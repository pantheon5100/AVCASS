# WPR calculation

Code for the wrong-prediction-rate (WPR) evaluation used for
audio source-separation outputs. It uses the PANNs
`Cnn14_DecisionLevelMax` sound-event detector to measure leakage between
speech, music, and sound-effect stems.

## Installation

Download and checksum the public PANNs checkpoint:

```bash
./download_checkpoint.sh
```


## Input layout

The evaluator recursively searches each method directory for `.wav` files.
It infers the ground-truth stem from the filename or a parent directory.
Supported names are:

- `speech`, `voice`, or `vocals`
- `music`
- `sfx`, `effects`, `sound_effect`, or `sound-effect`

For example:

```text
outputs/
  clip_0001/
    speech.wav
    music.wav
    sfx.wav
  clip_0002/
    speech.wav
    music.wav
    sfx.wav
```

## Run

Evaluate one or more methods:

```bash
./eval-AVDnR.sh \
  AVCASS=/path/to/avcass/outputs \
  mrx=/path/to/mrx/outputs \
  bandit=/path/to/bandit/outputs
```

Optional environment variables:

```bash
DEVICE=cuda:1 THRESHOLD=0.15 OUTPUT_DIR=./results ./eval-AVDnR.sh method=/path
```

CPU inference is selected automatically when CUDA is unavailable. The Python
entry point exposes the full interface:

```bash
python3 evaluate_wpr.py --help
```

## Outputs

The output directory contains:

- `wpr_pairs_threshold_*.csv`: mean WPR for every source-target pair.
- `wpr_by_source_threshold_*.csv`: mean of both wrong target predictions for
  each source type.
- `wpr_per_file_threshold_*.csv`: per-file values and denominator frame count.

For a source stem, WPR is the number of frames predicted as the wrong target
class divided by the number of frames predicted as any non-silent semantic
class. Semantic group probabilities are computed by taking the maximum PANNs
class logit in the group, applying sigmoid, and thresholding.

