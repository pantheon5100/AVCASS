<p align="center">
  <img src="assets/avcasslogo.png" alt="AV-CASS" width="140" />
</p>

<h1 align="center">Cinematic Audio Source Separation Using Visual Cues</h1>

<p align="center">
  Official code release for the CVPR paper <strong>Cinematic Audio Source Separation Using Visual Cues</strong>.
</p>

<p align="center">
  <strong>Kang Zhang<sup>1</sup>*, Suyeon Lee<sup>1</sup>*, Arda Senocak<sup>2</sup>+, Joon Son Chung<sup>1</sup>+</strong><br />
  <sup>1</sup> School of Electrical Engineering, KAIST<br />
  <sup>2</sup> Graduate School of Artificial Intelligence, UNIST
</p>

<p align="center">
  * equal contribution, + equal corresponding.
</p>

<p align="center">
  <a href="https://cass-flowmatching.github.io">Project Page</a> |
  <a href="https://arxiv.org/pdf/2603.26113v1">Paper</a>
</p>


## 🎬 Abstract

Cinematic Audio Source Separation (CASS) aims to decompose mixed film audio into speech, music, and sound effects, enabling applications like dubbing and remastering. Existing CASS approaches are audio-only, overlooking the inherent audio-visual nature of films, where sounds often align with visual cues. We present the first framework for audio-visual CASS (AV-CASS), leveraging visual context to enhance separation quality. Our method formulates CASS as a conditional generative modeling problem using conditional flow matching, enabling multimodal audio source separation. To address the lack of cinematic datasets with isolated sound tracks, we introduce a training data synthesis pipeline that pairs in-the-wild audio and video streams, such as facial videos for speech and scene videos for effects, and design a dedicated visual encoder for this dual-stream setup. Trained entirely on synthetic data, our model generalizes effectively to real-world cinematic content and achieves strong performance on synthetic, real-world, and audio-only CASS benchmarks.

## Updates

2026.08.05: We have released all files related to the Wrong Placement Ratio (WPR), including the class-label mapping and evaluation implementation.


## 📁 Repository Overview

This release contains two main components:

- `av_cass/`: training, inference, and evaluation code for audio-only and audio-visual CASS.
- `av_dnr/`: dataset preparation, manifest generation, and AVDnR synthesis utilities.

## ✅ Supported Workflows

- Stage-1 audio-only training.
- Stage-2 audio-visual training.
- Audio-only inference on AVDnR-format data.
- Audio-visual inference on AVDnR-format data.
- Objective evaluation on AVDnR outputs.
- Manifest-driven AVDnR generation.

## 📦 Pretrained weights

- AV-CASS checkpoint


  - Download the pretrained model weights from the links below.
  - You can jump to the Inference section (4. 🔍 Run inference) directly with these pretrained models.

| Model | Link |
| ------- | ----|
| Audio-only | [Google Drive Link](https://drive.google.com/file/d/1tK0HRQg3h-k-j612M8rxhGXhrN6T83fZ/view?usp=drive_link) |
| Audio-visual | [Google Drive Link](https://drive.google.com/file/d/1_d-RCP111No-wS-wrmxyK-zH87Sm2xzf/view?usp=drive_link) |

  - See `docs/CHECKPOINTS.md` for the expected layout.

- Visual backbone checkpoints for AV training and AV inference
  - CAVP checkpoint: at `Diff-Foley/diff_foley_ckpt/cavp_epoch66.ckpt` of [Diff-Foley](https://huggingface.co/SimianLuo/Diff-Foley)
  - TalkNet checkpoint: at [Google Drive](https://drive.google.com/file/d/1Qu1JC0zjrb_cc38LBOBD0lcol0d4Oy9y/view?usp=sharing)




## 🛠️ Installation

Create a Python environment with PyTorch, torchaudio, diffusers, accelerate, audioldm_eval, and the visual-backbone dependencies.

The minimal package list is provided in `requirements.txt`.

## 🚀 Quick Start

### 1. 🧱 Build source and split manifests

See `av_dnr/prepare_sources/README.md` for preparing source datasets (VGGSound & FMA).
After preparation, set the source directories in `configs/source_roots.json` to your paths.

```bash
cd av_dnr
python bin/build_source_manifest.py \
  --config configs/source_roots.json \
  --output-dir manifests/source_catalog

python bin/make_split_manifests.py \
  --catalog-dir manifests/source_catalog \
  --output-dir manifests/release_split \
  --seed 20260309 \
  --test-fraction 0.1

python bin/validate_split_manifests.py \
  --split-dir manifests/release_split
```

### 2. 🎧 Generate AVDnR

```bash
cd av_dnr
python bin/generate_dataset.py \
  --manifest-dir manifests/release_split \
  --output-root /path/to/output \
  --dataset-name AVDnR \
  --split test \
  --num-samples 1000 \
  --mixture-length 60
```

### 3. 🏋️ Train models

- Stage-1 audio-only training:

```bash
cd av_cass
DATASET_ROOT=/path/to/AVDnR \
RESULTS_DIR=/path/to/training_output \
NUM_PROCESSES=4 \
bash ./bin/train_ao.sh
```

- Stage-2 audio-visual training (requires a stage-1 checkpoint):

```bash
cd av_cass
DATASET_ROOT=/path/to/AVDnR \
RESULTS_DIR=/path/to/training_output \
INIT_CKPT=/path/to/ao_cass_checkpoint.pt \
CAVP_CKPT=/path/to/cavp.ckpt \
TALKNET_CKPT=/path/to/talknet.ckpt \
NUM_PROCESSES=4 \
bash ./bin/train_av.sh
```

### 4. 🔍 Run inference

- Audio-only:

```bash
cd av_cass
DATASET_ROOT=/path/to/AVDnR \
SAMPLE_DIR=/path/to/output_predictions \
CKPT=/path/to/ao_cass_checkpoint.pt \
NUM_GPUS=2 \
./bin/infer_ao.sh
```

- Audio-visual:

```bash
cd av_cass
DATASET_ROOT=/path/to/AVDnR \
SAMPLE_DIR=/path/to/output_predictions \
CKPT=/path/to/av_cass_checkpoint.pt \
CAVP_CKPT=/path/to/cavp.ckpt \
NUM_GPUS=2 \
./bin/infer_av.sh
```

### 5. 📊 Run evaluation

```bash
cd av_cass
PRED_ROOT=/path/to/output_predictions/run_name \
GT_ROOT=/path/to/AVDnR/test \
SYMLINK_ROOT=/path/to/tmp_symlinks \
RESULTS_OUT=/path/to/eval_results \
MODEL_NAME=AV-CASS \
./bin/evaluate.sh
```

## 📝 BibTex
```
@inproceedings{zhang2026cinematicaudiosourceseparation,
  title={Cinematic Audio Source Separation Using Visual Cues},
  author={Zhang, Kang and Lee, Suyeon and Senocak, Arda and Chung, Joon Son},
  booktitle={IEEE Conf. Comput. Vis. Pattern Recog.},
  year={2026}
}
```
