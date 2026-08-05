"""Minimal PANNs model definition needed by the WPR evaluation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram


def _init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        layer.bias.data.fill_(0.0)


def _init_bn(layer):
    layer.bias.data.fill_(0.0)
    layer.weight.data.fill_(1.0)


def _interpolate(x, ratio):
    batch_size, time_steps, classes_num = x.shape
    return (
        x[:, :, None, :]
        .repeat(1, 1, ratio, 1)
        .reshape(batch_size, time_steps * ratio, classes_num)
    )


def _pad_framewise_output(framewise_output, frames_num):
    if framewise_output.shape[1] >= frames_num:
        return framewise_output[:, :frames_num, :]
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1
    )
    return torch.cat((framewise_output, pad), dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        _init_layer(self.conv1)
        _init_layer(self.conv2)
        _init_bn(self.bn1)
        _init_bn(self.bn2)

    def forward(self, x, pool_size=(2, 2), pool_type="avg"):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "avg":
            return F.avg_pool2d(x, kernel_size=pool_size)
        if pool_type == "max":
            return F.max_pool2d(x, kernel_size=pool_size)
        if pool_type == "avg+max":
            return F.avg_pool2d(x, pool_size) + F.max_pool2d(x, pool_size)
        raise ValueError(f"Unknown pool type: {pool_type}")


class Cnn14DecisionLevelMax(nn.Module):
    """Cnn14_DecisionLevelMax from PANNs, including framewise logits."""

    def __init__(
        self,
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527,
    ):
        super().__init__()
        self.interpolate_ratio = 32
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )
        self.bn0 = nn.BatchNorm2d(mel_bins)
        # Attribute names intentionally match the public checkpoint state dict.
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes_num)
        _init_bn(self.bn0)
        _init_layer(self.fc1)
        _init_layer(self.fc_audioset)

    def forward(self, waveform):
        x = self.spectrogram_extractor(waveform)
        x = self.logmel_extractor(x)
        frames_num = x.shape[2]
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)

        for index in range(1, 7):
            block = getattr(self, f"conv_block{index}")
            pool_size = (1, 1) if index == 6 else (2, 2)
            x = block(x, pool_size=pool_size, pool_type="avg")
            x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        x = F.max_pool1d(x, 3, stride=1, padding=1) + F.avg_pool1d(
            x, 3, stride=1, padding=1
        )
        x = F.dropout(x, p=0.5, training=self.training).transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        segmentwise_logits = self.fc_audioset(x)
        segmentwise_output = torch.sigmoid(segmentwise_logits)
        return {
            "framewise_output": _pad_framewise_output(
                _interpolate(segmentwise_output, self.interpolate_ratio), frames_num
            ),
            "framewise_output_logits": _pad_framewise_output(
                _interpolate(segmentwise_logits, self.interpolate_ratio), frames_num
            ),
            "clipwise_output": torch.max(segmentwise_output, dim=1).values,
        }


def load_model(checkpoint_path, device, classes_num=527):
    model = Cnn14DecisionLevelMax(classes_num=classes_num)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
