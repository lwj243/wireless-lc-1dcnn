from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import LightweightEISNet, count_trainable_parameters

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "wireless_lc_dataset.npz"
OUTPUT = ROOT / "outputs"
STATE_NAMES = ["early", "active release", "post-release diffusion"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the wireless-LC lightweight 1D-CNN")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.8e-3)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--demo-delay", type=float, default=0.06)
    parser.add_argument("--no-live", action="store_true")
    parser.add_argument("--no-hold", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(
    features: np.ndarray,
    concentration: np.ndarray,
    state: np.ndarray,
    sample_ids: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features[indices]).float(),
        torch.from_numpy(concentration[indices]).float(),
        torch.from_numpy(state[indices]).long(),
        torch.from_numpy(sample_ids[indices]).long(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )


def class_weights(state: np.ndarray, indices: np.ndarray) -> torch.Tensor:
    counts = np.bincount(state[indices], minlength=3).astype(float)
    return torch.tensor(counts.sum() / (3.0 * counts), dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    regression_loss: nn.Module,
    classification_loss: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, np.ndarray | float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    count = 0
    true_c: list[np.ndarray] = []
    pred_c: list[np.ndarray] = []
    true_s: list[np.ndarray] = []
    pred_s: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    attentions: list[np.ndarray] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, concentration, state, sample_id in loader:
            x = x.to(device, non_blocking=True)
            concentration = concentration.to(device, non_blocking=True)
            state = state.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(x)
            reg = regression_loss(output["concentration"], concentration)
            cls = classification_loss(output["state_logits"], state)
            # State labels now describe operating phases, not concentration bins,
            # so concentration-state consistency is intentionally not imposed.
            loss = reg + 0.45 * cls
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            batch = x.shape[0]
            total_loss += float(loss.detach()) * batch
            count += batch
            true_c.append(concentration.detach().cpu().numpy())
            pred_c.append(output["concentration"].detach().cpu().numpy())
            true_s.append(state.detach().cpu().numpy())
            pred_s.append(output["state_logits"].argmax(1).detach().cpu().numpy())
            ids.append(sample_id.numpy())
            attentions.append(output["attention"].detach().cpu().numpy())
    true_c_array = np.concatenate(true_c)
    pred_c_array = np.concatenate(pred_c)
    true_s_array = np.concatenate(true_s)
    pred_s_array = np.concatenate(pred_s)
    return {
        "loss": total_loss / count,
        "mae": float(np.mean(np.abs(pred_c_array - true_c_array))),
        "rmse": float(np.sqrt(np.mean((pred_c_array - true_c_array) ** 2))),
        "accuracy": float(np.mean(pred_s_array == true_s_array)),
        "true_c": true_c_array,
        "pred_c": pred_c_array,
        "true_s": true_s_array,
        "pred_s": pred_s_array,
        "sample_id": np.concatenate(ids),
        "attention": np.concatenate(attentions),
    }


def confusion_matrix(true_state: np.ndarray, predicted_state: np.ndarray) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=int)
    for true_value, predicted_value in zip(true_state, predicted_state, strict=True):
        matrix[int(true_value), int(predicted_value)] += 1
    return matrix


def macro_f1(matrix: np.ndarray) -> float:
    scores = []
    for label in range(3):
        tp = matrix[label, label]
        fp = matrix[:, label].sum() - tp
        fn = matrix[label, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(2.0 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(scores))


class LivePlot:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.history = {
            key: []
            for key in [
                "train_loss",
                "validation_loss",
                "train_mae",
                "validation_mae",
                "train_accuracy",
                "validation_accuracy",
                "learning_rate",
            ]
        }
        if enabled:
            plt.ion()
        self.figure, axes = plt.subplots(2, 2, figsize=(11.6, 7.3))
        self.loss_axis, self.mae_axis, self.accuracy_axis, self.parity_axis = axes.ravel()
        self.figure.suptitle("Live training: COMSOL-informed wireless-LC 1D-CNN", fontsize=14)
        self.figure.canvas.manager.set_window_title("Wireless LC 1D-CNN Live Training")
        if enabled:
            plt.show(block=False)

    def update(
        self,
        epoch: int,
        train_metrics: dict[str, np.ndarray | float],
        validation_metrics: dict[str, np.ndarray | float],
        learning_rate: float,
    ) -> None:
        for split, metrics in [("train", train_metrics), ("validation", validation_metrics)]:
            for metric in ["loss", "mae", "accuracy"]:
                self.history[f"{split}_{metric}"].append(float(metrics[metric]))
        self.history["learning_rate"].append(learning_rate)
        epochs = np.arange(1, epoch + 1)

        self.loss_axis.clear()
        self.loss_axis.plot(epochs, self.history["train_loss"], color="#2474D2", label="Train")
        self.loss_axis.plot(epochs, self.history["validation_loss"], color="#D52270", label="Validation")
        self.loss_axis.set(title="Multi-task loss", xlabel="Epoch", ylabel="Loss")
        self.loss_axis.set_yscale("log")
        self.loss_axis.legend(frameon=False)

        self.mae_axis.clear()
        self.mae_axis.plot(epochs, self.history["train_mae"], color="#2474D2", label="Train")
        self.mae_axis.plot(epochs, self.history["validation_mae"], color="#D52270", label="Validation")
        self.mae_axis.set(title="Concentration decoding", xlabel="Epoch", ylabel="MAE (C/Cmax)")
        self.mae_axis.legend(frameon=False)

        self.accuracy_axis.clear()
        self.accuracy_axis.plot(
            epochs, 100 * np.asarray(self.history["train_accuracy"]), color="#2474D2", label="Train"
        )
        self.accuracy_axis.plot(
            epochs, 100 * np.asarray(self.history["validation_accuracy"]), color="#D52270", label="Validation"
        )
        self.accuracy_axis.set(
            title="Release-state classification", xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0, 102)
        )
        self.accuracy_axis.legend(frameon=False, loc="lower right")

        self.parity_axis.clear()
        colors = np.asarray(["#2474D2", "#1A9C89", "#D52270"])
        self.parity_axis.scatter(
            validation_metrics["true_c"],
            validation_metrics["pred_c"],
            c=colors[np.asarray(validation_metrics["true_s"], dtype=int)],
            s=18,
            edgecolor="white",
            linewidth=0.3,
        )
        self.parity_axis.plot([0, 1], [0, 1], "--", color="#687386", lw=1)
        self.parity_axis.set(
            title=f"Validation | epoch {epoch} | lr={learning_rate:.1e}",
            xlabel="True C/Cmax",
            ylabel="Predicted C/Cmax",
            xlim=(0, 1.02),
            ylim=(0, 1.02),
        )
        for axis in [self.loss_axis, self.mae_axis, self.accuracy_axis, self.parity_axis]:
            axis.grid(alpha=0.20)
        self.figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))
        if self.enabled:
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            plt.pause(0.001)

    def show_test(self, test: dict[str, np.ndarray | float], ood: dict[str, np.ndarray | float]) -> None:
        self.parity_axis.clear()
        self.parity_axis.scatter(
            test["true_c"], test["pred_c"], s=20, color="#2474D2", alpha=0.75, label="Held-out test"
        )
        self.parity_axis.scatter(
            ood["true_c"], ood["pred_c"], s=18, color="#E07A22", alpha=0.48, label="OOD stress test"
        )
        self.parity_axis.plot([0, 1], [0, 1], "--", color="#687386", lw=1)
        self.parity_axis.set(
            title=f"Test MAE={float(test['mae']):.3f}; OOD MAE={float(ood['mae']):.3f}",
            xlabel="True C/Cmax",
            ylabel="Predicted C/Cmax",
            xlim=(0, 1.02),
            ylim=(0, 1.02),
        )
        self.parity_axis.legend(frameon=False, loc="upper left")
        self.parity_axis.grid(alpha=0.20)
        self.figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.94))


def save_predictions(
    path: Path,
    split_name: str,
    metrics: dict[str, np.ndarray | float],
    device_id: np.ndarray,
    time_min: np.ndarray,
    concentration_um: np.ndarray,
    reference_cmax_um: float,
) -> None:
    sample_id = np.asarray(metrics["sample_id"], dtype=int)
    frame = pd.DataFrame(
        {
            "sample_id": sample_id,
            "split": split_name,
            "device_id": device_id[sample_id],
            "time_min": time_min[sample_id],
            "true_concentration_um": concentration_um[sample_id],
            "true_C_over_Cmax": metrics["true_c"],
            "predicted_C_over_Cmax": metrics["pred_c"],
            "predicted_concentration_um": np.asarray(metrics["pred_c"]) * reference_cmax_um,
            "true_state": np.asarray(STATE_NAMES)[np.asarray(metrics["true_s"], dtype=int)],
            "predicted_state": np.asarray(STATE_NAMES)[np.asarray(metrics["pred_s"], dtype=int)],
        }
    )
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = np.load(DATA, allow_pickle=False)
    raw_features = data["features"].astype(np.float32)
    concentration = data["concentration_norm"].astype(np.float32)
    state = data["state"].astype(np.int64)
    split_code = data["split_code"].astype(np.int64)
    sample_ids = np.arange(len(state), dtype=np.int64)
    indices = {
        name: np.flatnonzero(split_code == code) for code, name in enumerate(["train", "validation", "test", "ood"])
    }

    channel_mean = raw_features[indices["train"]].mean(axis=(0, 2), keepdims=True)
    channel_std = raw_features[indices["train"]].std(axis=(0, 2), keepdims=True)
    channel_std = np.maximum(channel_std, 1e-6)
    features = (raw_features - channel_mean) / channel_std
    np.savez(
        OUTPUT / "input_normalization.npz",
        channel_mean=channel_mean,
        channel_std=channel_std,
        channel_names=data["channel_names"],
        frequency_hz=data["frequency_hz"],
    )

    pin_memory = device.type == "cuda"
    loaders = {
        name: make_loader(
            features,
            concentration,
            state,
            sample_ids,
            split_indices,
            args.batch_size,
            shuffle=name == "train",
            pin_memory=pin_memory,
        )
        for name, split_indices in indices.items()
    }
    model = LightweightEISNet(input_channels=9, n_states=3).to(device)
    parameter_count = count_trainable_parameters(model)
    if parameter_count != 8981:
        raise RuntimeError(f"Expected 8,981 trainable parameters, got {parameter_count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.55, patience=8, min_lr=8e-6)
    regression_loss = nn.SmoothL1Loss(beta=0.035)
    classification_loss = nn.CrossEntropyLoss(
        weight=class_weights(state, indices["train"]).to(device), label_smoothing=0.025
    )
    live_plot = LivePlot(enabled=not args.no_live)
    best_loss = float("inf")
    best_epoch = 0
    start = time.perf_counter()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Samples: train={len(indices['train'])}, "
        f"validation={len(indices['validation'])}, "
        f"test={len(indices['test'])}, OOD={len(indices['ood'])}"
    )
    print(f"Trainable parameters: {parameter_count:,}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], device, regression_loss, classification_loss, optimizer)
        validation_metrics = run_epoch(model, loaders["validation"], device, regression_loss, classification_loss, None)
        scheduler.step(float(validation_metrics["loss"]))
        learning_rate = optimizer.param_groups[0]["lr"]
        if float(validation_metrics["loss"]) < best_loss:
            best_loss = float(validation_metrics["loss"])
            best_epoch = epoch
            torch.save(
                {
                    "format_version": 1,
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "parameter_count": parameter_count,
                    "input_channels": 9,
                    "n_states": 3,
                    "channel_mean": torch.from_numpy(channel_mean),
                    "channel_std": torch.from_numpy(channel_std),
                    "frequency_hz": torch.as_tensor(data["frequency_hz"]),
                    "channel_names": data["channel_names"].tolist(),
                    "state_names": STATE_NAMES,
                },
                OUTPUT / "best_wireless_lc_1dcnn.pt",
            )
        live_plot.update(epoch, train_metrics, validation_metrics, learning_rate)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            train_loss = float(train_metrics["loss"])
            validation_loss = float(validation_metrics["loss"])
            train_mae = float(train_metrics["mae"])
            validation_mae = float(validation_metrics["mae"])
            train_accuracy = 100 * float(train_metrics["accuracy"])
            validation_accuracy = 100 * float(validation_metrics["accuracy"])
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"loss {train_loss:.4f}/{validation_loss:.4f} | "
                f"MAE {train_mae:.3f}/{validation_mae:.3f} | "
                f"accuracy {train_accuracy:.1f}/{validation_accuracy:.1f}%"
            )
        if args.demo_delay > 0:
            time.sleep(args.demo_delay)

    checkpoint = torch.load(OUTPUT / "best_wireless_lc_1dcnn.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    validation_metrics = run_epoch(model, loaders["validation"], device, regression_loss, classification_loss, None)
    test_metrics = run_epoch(model, loaders["test"], device, regression_loss, classification_loss, None)
    ood_metrics = run_epoch(model, loaders["ood"], device, regression_loss, classification_loss, None)
    live_plot.show_test(test_metrics, ood_metrics)
    live_plot.figure.savefig(OUTPUT / "live_training_wireless.png", dpi=250, bbox_inches="tight")

    reference_cmax_um = float(data["reference_cmax_um"])
    for split_name, metrics in [("validation", validation_metrics), ("test", test_metrics), ("ood", ood_metrics)]:
        save_predictions(
            OUTPUT / f"{split_name}_predictions.csv",
            split_name,
            metrics,
            data["device_id"],
            data["time_min"],
            data["concentration_um"],
            reference_cmax_um,
        )
    np.savez_compressed(
        OUTPUT / "test_attention.npz",
        attention=test_metrics["attention"],
        sample_id=test_metrics["sample_id"],
        encoded_frequency_hz=data["frequency_hz"].reshape(2, 32).mean(axis=0),
    )
    matrices = {
        name: confusion_matrix(np.asarray(metrics["true_s"]), np.asarray(metrics["pred_s"]))
        for name, metrics in [("test", test_metrics), ("ood", ood_metrics)]
    }
    metrics_json = {
        "data_source": (
            "Measured-self-discharge-gated COMSOL transport plus FE-informed "
            "virtual wireless LC spectra; not experimental wireless measurements"
        ),
        "frequency_range_mhz": [11.5, 15.5],
        "circuit_points": 1001,
        "network_points": 64,
        "input_channels": 9,
        "baseline_calibration_required": True,
        "trainable_parameters": parameter_count,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "seed": args.seed,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - start,
        "split_by_virtual_device": True,
        "test": {
            "mae_C_over_Cmax": float(test_metrics["mae"]),
            "rmse_C_over_Cmax": float(test_metrics["rmse"]),
            "mae_um": float(test_metrics["mae"]) * reference_cmax_um,
            "accuracy": float(test_metrics["accuracy"]),
            "macro_f1": macro_f1(matrices["test"]),
            "confusion_matrix": matrices["test"].tolist(),
        },
        "ood": {
            "mae_C_over_Cmax": float(ood_metrics["mae"]),
            "rmse_C_over_Cmax": float(ood_metrics["rmse"]),
            "mae_um": float(ood_metrics["mae"]) * reference_cmax_um,
            "accuracy": float(ood_metrics["accuracy"]),
            "macro_f1": macro_f1(matrices["ood"]),
            "confusion_matrix": matrices["ood"].tolist(),
        },
        "history": live_plot.history,
    }
    (OUTPUT / "wireless_training_metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    print(json.dumps({"test": metrics_json["test"], "ood": metrics_json["ood"]}, indent=2))
    print("WIRELESS_LC_TRAINING=PASS")
    if not args.no_live and not args.no_hold:
        print("Training complete. Close the figure window to return to the terminal.")
        plt.ioff()
        plt.show()
    else:
        plt.close(live_plot.figure)


if __name__ == "__main__":
    main()
