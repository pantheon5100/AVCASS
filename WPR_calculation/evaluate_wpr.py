#!/usr/bin/env python3
"""Calculate wrong-prediction rates (WPR) for separated audio tracks."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from model import load_model


CLASS_NAMES = ("speech", "music", "sound effect")
PAIR_NAMES = tuple(
    f"{source}-{target}"
    for source in CLASS_NAMES
    for target in CLASS_NAMES
    if source != target
)
SOURCE_ALIASES = {
    "speech": "speech",
    "vocals": "speech",
    "voice": "speech",
    "music": "music",
    "sfx": "sound effect",
    "effects": "sound effect",
    "sound_effect": "sound effect",
    "sound-effect": "sound effect",
}


def parse_method(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--method must have the form NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--method must have a non-empty name and path")
    return name.strip(), Path(path).expanduser()


def read_class_map(csv_path):
    groups = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups[row["main_class"].strip()].append(int(row["index"]))
    missing = set(CLASS_NAMES) - groups.keys()
    if missing:
        raise ValueError(f"Class map is missing groups: {sorted(missing)}")
    return dict(groups)


def infer_source(path):
    candidates = [path.stem.lower(), *(part.lower() for part in reversed(path.parts))]
    for candidate in candidates:
        normalized = candidate.replace(" ", "_")
        if normalized in SOURCE_ALIASES:
            return SOURCE_ALIASES[normalized]
    return None


def load_audio(path, sample_rate):
    waveform, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if source_rate != sample_rate:
        divisor = math.gcd(source_rate, sample_rate)
        waveform = resample_poly(
            waveform, sample_rate // divisor, source_rate // divisor
        ).astype(np.float32)
    if waveform.size < 1024:
        waveform = np.pad(waveform, (0, 1024 - waveform.size))
    return waveform


def predict_groups(path, model, device, class_map, threshold, sample_rate):
    waveform = torch.from_numpy(load_audio(path, sample_rate))[None, :].to(device)
    with torch.inference_mode():
        logits = model(waveform)["framewise_output_logits"][0]

    group_predictions = []
    for name in (*CLASS_NAMES, "Silence"):
        indices = class_map[name]
        # This preserves the original evaluation: maximum class logit per frame,
        # followed by sigmoid and a binary threshold.
        group_probability = torch.sigmoid(logits[:, indices].max(dim=1).values)
        group_predictions.append(group_probability >= threshold)
    return torch.stack(group_predictions).cpu().numpy()


def evaluate_method(name, root, model, device, class_map, args):
    if not root.is_dir():
        raise FileNotFoundError(f"Method directory does not exist: {root}")

    pair_values = {pair: [] for pair in PAIR_NAMES}
    file_rows = []
    wav_paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"
    )
    if not wav_paths:
        raise ValueError(f"No WAV files found under {root}")

    for path in wav_paths:
        source = infer_source(path.relative_to(root))
        if source is None:
            print(f"[skip] Cannot infer track type from: {path}")
            continue

        predictions = predict_groups(
            path,
            model,
            device,
            class_map,
            args.threshold,
            args.sample_rate,
        )
        non_silence = predictions[:3].any(axis=0)
        denominator = int(non_silence.sum())
        if denominator == 0:
            print(f"[skip] No predicted non-silent frames: {path}")
            continue

        row = {"method": name, "file": str(path), "source": source, "frames": denominator}
        for target_index, target in enumerate(CLASS_NAMES):
            if target == source:
                continue
            pair = f"{source}-{target}"
            value = float(predictions[target_index].sum() / denominator)
            pair_values[pair].append(value)
            row[target] = value
            print(f"{name}: {path.name}: {source} -> {target}: {value:.4f}")
        file_rows.append(row)

    if not file_rows:
        raise ValueError(f"No recognizable, non-silent source tracks found under {root}")
    return pair_values, file_rows


def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Calculate WPR for speech, music, and sound-effect stems."
    )
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        type=parse_method,
        metavar="NAME=PATH",
        help="Method name and root containing WAV stems; repeat for multiple methods.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--class-map",
        type=Path,
        default=project_dir / "metadata" / "class_labels_with_main_class.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=project_dir / "results")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--sample-rate", type=int, default=32000)
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda', or a device such as 'cuda:1' (default: auto).",
    )
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    method_names = [name for name, _ in args.method]
    if len(method_names) != len(set(method_names)):
        parser.error("each --method NAME must be unique")
    return args


def main():
    args = parse_args()
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device}")

    class_map = read_class_map(args.class_map)
    model = load_model(args.checkpoint, device, classes_num=sum(map(len, class_map.values())))

    all_metrics = {}
    all_file_rows = []
    for name, root in args.method:
        pair_values, file_rows = evaluate_method(
            name, root, model, device, class_map, args
        )
        all_metrics[name] = pair_values
        all_file_rows.extend(file_rows)

    pair_rows = []
    source_rows = []
    for method, metrics in all_metrics.items():
        pair_rows.append(
            {"method": method, **{pair: mean_or_nan(metrics[pair]) for pair in PAIR_NAMES}}
        )
        source_rows.append(
            {
                "method": method,
                **{
                    source: mean_or_nan(
                        [
                            value
                            for pair, values in metrics.items()
                            if pair.startswith(source + "-")
                            for value in values
                        ]
                    )
                    for source in CLASS_NAMES
                },
            }
        )

    suffix = f"{args.threshold:g}"
    write_csv(
        args.output_dir / f"wpr_pairs_threshold_{suffix}.csv",
        ["method", *PAIR_NAMES],
        pair_rows,
    )
    write_csv(
        args.output_dir / f"wpr_by_source_threshold_{suffix}.csv",
        ["method", *CLASS_NAMES],
        source_rows,
    )
    write_csv(
        args.output_dir / f"wpr_per_file_threshold_{suffix}.csv",
        ["method", "file", "source", "frames", *CLASS_NAMES],
        all_file_rows,
    )
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
