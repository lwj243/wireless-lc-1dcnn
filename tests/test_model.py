from __future__ import annotations

import torch

from wireless_lc_1dcnn import LightweightEISNet, count_trainable_parameters


def test_model_shapes_and_parameter_count() -> None:
    model = LightweightEISNet(input_channels=9, n_states=3).eval()
    output = model(torch.zeros(2, 9, 64))
    assert output["concentration"].shape == (2,)
    assert output["state_logits"].shape == (2, 3)
    assert output["attention"].shape == (2, 32)
    assert count_trainable_parameters(model) == 8_981


def test_attention_is_normalized() -> None:
    model = LightweightEISNet(input_channels=9, n_states=3).eval()
    with torch.inference_mode():
        attention = model(torch.randn(3, 9, 64))["attention"]
    assert torch.allclose(attention.sum(dim=1), torch.ones(3), atol=1e-6)
