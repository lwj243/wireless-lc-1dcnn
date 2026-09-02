from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import LightweightEISNet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "wireless_lc_1dcnn_weights.pt"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "wireless_lc_dataset.npz"
STATE_NAMES = ("early", "active release", "post-release diffusion")


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a tensor-only checkpoint without permitting general pickle execution."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required = {"format_version", "model_state", "input_channels", "n_states", "channel_mean", "channel_std"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {sorted(missing)}")
    if checkpoint["format_version"] != 1:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint['format_version']!r}")
    if checkpoint["input_channels"] != 9 or checkpoint["n_states"] != 3:
        raise ValueError("Checkpoint architecture does not match the published model")
    if not isinstance(checkpoint["model_state"], dict):
        raise TypeError("model_state must be a dictionary")
    if not all(isinstance(value, torch.Tensor) for value in checkpoint["model_state"].values()):
        raise TypeError("model_state may contain tensors only")
    if tuple(checkpoint["channel_mean"].shape) != (1, 9, 1):
        raise ValueError("Unexpected channel_mean shape")
    if tuple(checkpoint["channel_std"].shape) != (1, 9, 1):
        raise ValueError("Unexpected channel_std shape")
    return checkpoint


def build_model(checkpoint: dict[str, Any], device: torch.device) -> LightweightEISNet:
    model = LightweightEISNet(
        input_channels=int(checkpoint["input_channels"]),
        n_states=int(checkpoint["n_states"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def predict(features: np.ndarray, checkpoint_path: Path = DEFAULT_CHECKPOINT) -> dict[str, Any]:
    device = torch.device("cpu")
    checkpoint = load_checkpoint(checkpoint_path, device)
    if features.shape != (9, 64):
        raise ValueError(f"Expected features with shape (9, 64), received {features.shape}")
    mean = checkpoint["channel_mean"].cpu().numpy()
    std = checkpoint["channel_std"].cpu().numpy()
    normalized = (features.astype(np.float32, copy=False)[None, ...] - mean) / std
    model = build_model(checkpoint, device)
    with torch.inference_mode():
        output = model(torch.from_numpy(normalized).float())
    state_index = int(output["state_logits"].argmax(dim=1).item())
    return {
        "concentration_C_over_Cmax": float(output["concentration"].item()),
        "state_index": state_index,
        "state": STATE_NAMES[state_index],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run safe inference on one bundled synthetic sample")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.dataset, allow_pickle=False) as dataset:
        features = dataset["features"]
        if not 0 <= args.sample_index < len(features):
            raise IndexError(f"sample-index must be between 0 and {len(features) - 1}")
        result = predict(features[args.sample_index], args.checkpoint)
        result["sample_index"] = args.sample_index
        result["true_concentration_C_over_Cmax"] = float(dataset["concentration_norm"][args.sample_index])
        result["true_state_index"] = int(dataset["state"][args.sample_index])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
