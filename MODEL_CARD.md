# Model card: Wireless-LC 1D-CNN

## Intended use

Research-only analysis of simulated wireless LC spectra. The model estimates normalized sensor-region concentration and classifies three simulated operating phases.

## Out-of-scope use

- clinical, diagnostic, therapeutic, or safety-critical decisions
- claims of experimental, in-vivo, or implant validation
- deployment without sensor calibration and drift testing
- inference on spectra with an unknown channel order or frequency grid

## Architecture

The network uses a convolutional stem, multi-scale depthwise-separable frequency blocks, attention pooling, and two output heads. It accepts arrays shaped `(batch, 9, 64)` and has 8,981 trainable parameters.

## Data

The bundled dataset contains 3,750 synthetic samples from 150 virtual devices. Splits are grouped by device to prevent the same virtual device from appearing in more than one split. The spectra combine measured-self-discharge-gated transport states with COMSOL-FE-informed circuit parameters; they are not measured wireless spectra.

## Reported performance

| Evaluation | Concentration MAE | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Held-out virtual devices | 0.00238 µM | 0.955 | 0.963 |
| OOD stress test | 0.01058 µM | 0.761 | 0.786 |

The OOD test deliberately widens tissue, component, calibration, kinetic, coupling, and measurement ranges.

## Limitations

- The sensor transfer function is not experimentally calibrated.
- The nominal resonance shift is approximately −96 kHz, about 1.5 bins on the 64-point wide sweep and approximately 0.12 resonance linewidth.
- Baseline drift, temperature, alignment, tissue variability, and ex-vivo/in-vivo performance remain unvalidated.
- A simple physics-based resonance baseline should remain part of future comparative evaluation.

## Security

The distributed checkpoint contains tensors and basic metadata only. Load it through `wireless_lc_1dcnn.inference.load_checkpoint`, which uses restricted PyTorch loading and validates its structure. Do not substitute arbitrary third-party checkpoints.

