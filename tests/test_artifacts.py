from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from wireless_lc_1dcnn.inference import build_model, load_checkpoint, predict

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "wireless_lc_1dcnn_weights.pt"
DATASET = ROOT / "data" / "wireless_lc_dataset.npz"


def test_checkpoint_uses_restricted_loader() -> None:
    checkpoint = load_checkpoint(CHECKPOINT, torch.device("cpu"))
    model = build_model(checkpoint, torch.device("cpu"))
    assert model.training is False


def test_dataset_contains_numeric_arrays_only() -> None:
    with np.load(DATASET, allow_pickle=False) as dataset:
        assert dataset["features"].shape[1:] == (9, 64)
        assert all(dataset[name].dtype.kind != "O" for name in dataset.files)


def test_bundled_sample_inference() -> None:
    with np.load(DATASET, allow_pickle=False) as dataset:
        result = predict(dataset["features"][0], CHECKPOINT)
    assert 0.0 <= result["concentration_C_over_Cmax"] <= 1.0
    assert result["state_index"] in {0, 1, 2}
