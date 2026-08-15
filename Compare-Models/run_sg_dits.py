"""Train SG-DiTs on an immutable version3 benchmark."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACKAGE = HERE / "SG-DiTs"
for path in (ROOT, PACKAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from comparison_protocol import (  # noqa: E402
    DEFAULT_SEEDS, PROTOCOLS, evaluate_predictions, load_summary,
    protocol_manifest, seed_everything, write_csv, write_json,
)
from features import FEATURE_NAMES, TJU_SOURCE_FEATURES, extract_sg_dits_features  # noqa: E402
from model import ConditionalDiT, DiffusionSchedule  # noqa: E402


@dataclass(frozen=True)
class Config:
    learning_rate: float = 1e-4
    epochs: int = 200
    batch_size: int = 256
    dimension: int = 256
    heads: int = 8
    depth: int = 12
    patch_size: int = 4
    diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    # The paper obtains the predictive distribution through multiple
    # samplings but does not disclose the count. Keep the original
    # implementation choice of five samples rather than tuning this count.
    samples: int = 5
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-6
    refit_full_training: bool = True


class ForecastWindows(Dataset):
    def __init__(self, frames, features, samples, sequence_length,
                 feature_mean, feature_std, rated_capacity,
                 capacity_mean, capacity_std):
        self.frames = frames
        self.features = features
        self.samples = samples
        self.sequence_length = sequence_length
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.rated_capacity = rated_capacity
        self.capacity_mean = float(capacity_mean)
        self.capacity_std = float(capacity_std)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, target = self.samples[index]
        begin = target - self.sequence_length
        health = (self.features[name][begin:target] - self.feature_mean) / self.feature_std
        capacity = self.frames[name]["Capacity"].to_numpy(dtype=np.float32) / self.rated_capacity
        # Direct capacity trajectory: no previous-capacity residual adapter.
        clean = (capacity[begin + 1:target + 1] - self.capacity_mean) / self.capacity_std
        history = (capacity[begin:target] - self.capacity_mean) / self.capacity_std
        condition_soh = capacity[target - 1]
        return (
            torch.as_tensor(health, dtype=torch.float32),
            torch.as_tensor(clean, dtype=torch.float32),
            torch.as_tensor(history, dtype=torch.float32),
            torch.as_tensor(condition_soh, dtype=torch.float32),
            torch.as_tensor(self.frames[name]["Cycle"].iloc[target], dtype=torch.long),
        )


def _split(length: int) -> int:
    return int(math.floor(length * 0.8))


def build_data(dataset: str, seed: int):
    protocol = PROTOCOLS[dataset]
    frames = load_summary(protocol)
    raw_directories = {
        "nasa": ROOT / "data/version3/NASA data",
        "calce": ROOT / "data/version3/CALCE data",
        "tju": ROOT / "data/version3/TJU data/Dataset_3_NCM_NCA_battery",
    }
    feature_frames = extract_sg_dits_features(dataset, raw_directories[dataset], frames)
    values = {
        name: frame[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
        for name, frame in feature_frames.items()
    }
    test_name = protocol.batteries[0]
    train_names = [name for name in protocol.batteries if name != test_name]
    train_samples, validation_samples = [], []
    for name in train_names:
        split = _split(len(frames[name]))
        train_samples.extend((name, target) for target in range(protocol.seq_len, split))
        validation_samples.extend((name, target) for target in range(split, len(frames[name])))
    fit = np.concatenate([values[name][:_split(len(frames[name]))] for name in train_names])
    mean, std = fit.mean(axis=0), fit.std(axis=0)
    std[std < 1e-8] = 1.0
    capacity_fit = np.concatenate([
        frames[name]["Capacity"].to_numpy(dtype=np.float32)[:_split(len(frames[name]))]
        / protocol.rated_capacity
        for name in train_names
    ])
    capacity_mean = float(capacity_fit.mean())
    capacity_std = float(capacity_fit.std())
    if capacity_std < 1e-8:
        capacity_std = 1.0
    test_samples = [(test_name, target) for target in range(protocol.seq_len, len(frames[test_name]))]
    datasets = [
        ForecastWindows(frames, values, samples, protocol.seq_len, mean, std,
                        protocol.rated_capacity, capacity_mean, capacity_std)
        for samples in (train_samples, validation_samples, test_samples)
    ]
    full_samples = [
        (name, target) for name in train_names
        for target in range(protocol.seq_len, len(frames[name]))
    ]
    full_fit = np.concatenate([values[name] for name in train_names])
    full_mean, full_std = full_fit.mean(axis=0), full_fit.std(axis=0)
    full_std[full_std < 1e-8] = 1.0
    full_capacity_fit = np.concatenate([
        frames[name]["Capacity"].to_numpy(dtype=np.float32) / protocol.rated_capacity
        for name in train_names
    ])
    full_capacity_mean = float(full_capacity_fit.mean())
    full_capacity_std = float(full_capacity_fit.std())
    if full_capacity_std < 1e-8:
        full_capacity_std = 1.0
    full_datasets = [
        ForecastWindows(frames, values, samples, protocol.seq_len, full_mean, full_std,
                        protocol.rated_capacity, full_capacity_mean, full_capacity_std)
        for samples in (full_samples, test_samples)
    ]
    return (
        protocol, frames, train_names, datasets, mean, std, capacity_mean, capacity_std,
        full_datasets, full_mean, full_std, full_capacity_mean, full_capacity_std,
    )


def _move(batch, device):
    return tuple(value.to(device) for value in batch)


@torch.no_grad()
def validation_loss(model, diffusion, loader, device, seed):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed + 100003)
    total, count = 0.0, 0
    for health, clean, capacity_history, soh, _ in loader:
        health, clean, capacity_history, soh = (
            value.to(device) for value in (health, clean, capacity_history, soh)
        )
        timestep = torch.randint(0, diffusion.steps, (len(clean),), device=device, generator=generator)
        noise = torch.randn(clean.shape, device=device, generator=generator)
        noisy = diffusion.add_noise(clean, timestep, noise)
        prediction, _ = model(noisy, timestep, health, soh, capacity_history)
        loss = torch.sqrt(torch.mean((prediction - noise) ** 2) + 1e-12)
        total += float(loss) * len(clean)
        count += len(clean)
    return total / max(count, 1)


def fit(model, diffusion, training, validation, config, device, seed, tag):
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(training, batch_size=config.batch_size, shuffle=True,
                              generator=generator, num_workers=0)
    validation_loader = DataLoader(validation, batch_size=config.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_state, best_loss, best_epoch = None, float("inf"), 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for health, clean, capacity_history, soh, _ in train_loader:
            health, clean, capacity_history, soh = (
                value.to(device) for value in (health, clean, capacity_history, soh)
            )
            timestep = torch.randint(0, diffusion.steps, (len(clean),), device=device)
            noise = torch.randn_like(clean)
            noisy = diffusion.add_noise(clean, timestep, noise)
            predicted_noise, _ = model(noisy, timestep, health, soh, capacity_history)
            # The paper explicitly names RMSE, rather than the usual DDPM MSE.
            loss = torch.sqrt(torch.mean((predicted_noise - noise) ** 2) + 1e-12)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(clean)
            count += len(clean)
        train_rmse = total / max(count, 1)
        # Fixed validation noise makes early-stopping comparisons deterministic
        # across epochs; only the learned parameters change.
        val_rmse = validation_loss(model, diffusion, validation_loader, device, seed)
        history.append({"epoch": epoch, "train_noise_rmse": train_rmse,
                        "validation_noise_rmse": val_rmse})
        print(f"{tag} epoch={epoch:03d}/{config.epochs} train={train_rmse:.7f} val={val_rmse:.7f}", flush=True)
        if val_rmse < best_loss - config.early_stopping_min_delta:
            best_state, best_loss, best_epoch = copy.deepcopy(model.state_dict()), val_rmse, epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(
                    f"{tag} early_stop epoch={epoch:03d} best_epoch={best_epoch:03d} "
                    f"best_val={best_loss:.7f} patience={config.early_stopping_patience}",
                    flush=True,
                )
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return history, best_epoch, best_loss


def fit_fixed_epochs(model, diffusion, training, config, device, seed, epochs, tag):
    """Refit on all non-test-cell windows for the inner-selected duration."""
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(training, batch_size=config.batch_size, shuffle=True,
                        generator=generator, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for health, clean, capacity_history, soh, _ in loader:
            health, clean, capacity_history, soh = (
                value.to(device) for value in (health, clean, capacity_history, soh)
            )
            timestep = torch.randint(0, diffusion.steps, (len(clean),), device=device)
            noise = torch.randn_like(clean)
            noisy = diffusion.add_noise(clean, timestep, noise)
            predicted_noise, _ = model(noisy, timestep, health, soh, capacity_history)
            loss = torch.sqrt(torch.mean((predicted_noise - noise) ** 2) + 1e-12)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(clean)
            count += len(clean)
        train_rmse = total / max(count, 1)
        history.append({"epoch": epoch, "train_noise_rmse": train_rmse})
        print(f"{tag} epoch={epoch:03d}/{epochs} train={train_rmse:.7f}", flush=True)
    return history


@torch.no_grad()
def predict(model, diffusion, dataset, config, device, seed, rated_capacity,
             capacity_mean, capacity_std):
    inference_batch_size = 256
    loader = DataLoader(dataset, batch_size=inference_batch_size, shuffle=False, num_workers=0)
    means, standard_deviations, targets, cycles = [], [], [], []
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed + 200003)
    for health, clean, capacity_history, soh, cycle in loader:
        health, clean, capacity_history, soh = (
            value.to(device) for value in (health, clean, capacity_history, soh)
        )
        repeated_health = health.repeat_interleave(config.samples, dim=0)
        repeated_history = capacity_history.repeat_interleave(config.samples, dim=0)
        repeated_soh = soh.repeat_interleave(config.samples, dim=0)
        generated = diffusion.sample(
            model, repeated_health, repeated_soh, generator, history=repeated_history
        )
        final = generated[:, -1].reshape(len(health), config.samples)
        final = final * capacity_std + capacity_mean
        means.append(final.mean(dim=1).cpu().numpy() * rated_capacity)
        standard_deviations.append(final.std(dim=1, unbiased=False).cpu().numpy() * rated_capacity)
        targets.append((clean[:, -1].cpu().numpy() * capacity_std + capacity_mean)
                       * rated_capacity)
        cycles.append(cycle.numpy())
    return tuple(np.concatenate(item) for item in (targets, means, standard_deviations, cycles))


def run(dataset, seed, output_root, device, max_epochs=None, samples=None,
        checkpoint_path=None, prepared_data=None):
    protocol = PROTOCOLS[dataset]
    test_name = protocol.batteries[0]
    output = output_root / "sg-dits" / dataset / test_name / f"seed_{seed}"
    result_path = output / "results.json"
    config = Config(
        epochs=int(max_epochs or 200), samples=int(samples or 5),
    )
    if result_path.exists():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("status") == "complete" and existing.get("config") == asdict(config):
                print(f"skip complete: {output}", flush=True)
                return
        except (OSError, json.JSONDecodeError):
            pass
    seed_everything(seed)
    if prepared_data is None:
        prepared_data = build_data(dataset, seed)
    (
        protocol, frames, train_names, datasets, mean, std, capacity_mean, capacity_std,
        full_datasets, full_mean, full_std, full_capacity_mean, full_capacity_std,
    ) = prepared_data
    training, validation, testing = datasets
    full_training, full_testing = full_datasets
    model = ConditionalDiT(
        protocol.seq_len, len(FEATURE_NAMES), config.patch_size,
        config.dimension, config.heads, config.depth,
    ).to(device)
    diffusion = DiffusionSchedule(config.diffusion_steps, config.beta_start, config.beta_end).to(device)
    started = time.time()
    if checkpoint_path is None:
        inner_history, best_epoch, best_validation = fit(
            model, diffusion, training, validation, config, device, seed,
            f"[sg-dits/{dataset}/{test_name}/seed={seed}]",
        )
        seed_everything(seed)
        model = ConditionalDiT(
            protocol.seq_len, len(FEATURE_NAMES), config.patch_size,
            config.dimension, config.heads, config.depth,
        ).to(device)
        diffusion = DiffusionSchedule(
            config.diffusion_steps, config.beta_start, config.beta_end
        ).to(device)
        history = fit_fixed_epochs(
            model, diffusion, full_training, config, device, seed, best_epoch,
            f"[sg-dits-refit/{dataset}/{test_name}/seed={seed}]",
        )
        testing = full_testing
        capacity_mean, capacity_std = full_capacity_mean, full_capacity_std
        mean, std = full_mean, full_std
        checkpoint_source = "trained with inner epoch selection and full non-test-cell refit"
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        prior = checkpoint.get("result", {})
        history, inner_history = [], []
        best_epoch = prior.get("best_epoch")
        best_validation = prior.get("best_validation_noise_rmse")
        checkpoint_source = str(Path(checkpoint_path).resolve())
    y_true, y_pred, y_std, cycles = predict(
        model, diffusion, testing, config, device, seed, protocol.rated_capacity,
        capacity_mean, capacity_std,
    )
    metrics, prediction_rows = evaluate_predictions(cycles, y_true, y_pred, protocol)
    for row, std_value in zip(prediction_rows, y_std):
        row.update({"capacity_std": float(std_value), "capacity_p05": row["capacity_pred"] - 1.645 * float(std_value),
                    "capacity_p95": row["capacity_pred"] + 1.645 * float(std_value),
                    "model": "sg-dits", "dataset": dataset, "battery": test_name, "seed": seed})
    result = {
        "status": "complete", "model": "sg-dits", "dataset": dataset,
        "paper_doi": "10.1016/j.est.2026.120479", "test_battery": test_name,
        "train_batteries": train_names, "seed": seed,
        "task": f"{protocol.seq_len}-cycle rolling one-step capacity prediction",
        "metrics": metrics, "best_epoch": best_epoch,
        "best_validation_noise_rmse": best_validation,
        "epochs_trained": len(history),
        "inner_epochs_trained": len(inner_history),
        "stopped_early": len(history) < config.epochs,
        "elapsed_seconds": time.time() - started, "config": asdict(config),
        "inference_batch_size": 256,
        "checkpoint_source": checkpoint_source,
        "features": list(FEATURE_NAMES),
        "start_points": list(protocol.start_points),
        "feature_extraction": {
            "savitzky_golay_window": 11,
            "savitzky_golay_polyorder": 3,
            "shannon_entropy_bins": 32,
            "maximum_lyapunov_exponent": "Rosenstein-style nearest-neighbour trajectory-divergence slope",
        },
        "normalization": {"type": "all non-test-cell standardization after inner epoch selection; capacity/rated_capacity",
                          "mean": mean.tolist(), "std": std.tolist(),
                          "capacity_mean": capacity_mean,
                          "capacity_std": capacity_std},
        "processed_summary": protocol.summary,
        "raw_source": f"read-only raw {dataset.upper()} records under data/version3",
        "adaptations": [
            f"paper window 40 changed to unified {dataset.upper()} window {protocol.seq_len}",
            f"paper epochs undisclosed; {dataset.upper()} maximum budget fixed at 200",
            f"held-out {test_name} excluded from normalization and training",
            "training duration selected by inner 80/20 validation, then refitted from scratch on all non-test-cell windows",
            f"clean diffusion trajectory is the direct length-{protocol.seq_len} capacity sequence standardized by training-only global statistics",
            "observed last-cycle SOH replaces unavailable future SOH as the leakage-safe AdaLN condition",
            "paper EOL threshold 80% changed to the repository-wide 70% EOL protocol",
            "paper variance-head supervision is undisclosed; sampling uses the DDPM posterior variance",
            "default inference uses unconstrained reverse diffusion; the previous non-paper inpainting path is disabled",
            "the complete observed capacity window enters the paper's input/embedding stage as a conditioning sequence; output remains direct capacity",
        ],
    }
    if dataset == "calce":
        result["adaptations"].append(
            "CALCE first reproduces the documented 40-cycle two-sigma filtering, then resolves the legacy row-count difference by normalized chronological progress without capacity-label matching"
        )
    elif dataset == "tju":
        result["adaptations"].append(
            "version3 TJU provides per-cycle charge statistics rather than sampled curves; twelve disclosed charge indicators condition the unchanged SG-DiTs architecture"
        )
        result["tju_condition_source_features"] = list(TJU_SOURCE_FEATURES)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "result": result}, output / "checkpoint.pt")
    write_csv(output / "training_history.csv", history)
    write_csv(output / "inner_validation_history.csv", inner_history)
    write_csv(output / "predictions.csv", prediction_rows)
    write_json(result_path, result)
    print(f"complete: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(PROTOCOLS), default="nasa")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/comparison_models_10seeds")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=None,
        help="Root containing sg-dits/<dataset>/<battery>/seed_<seed>/checkpoint.pt",
    )
    args = parser.parse_args()
    if args.checkpoint_path is not None and args.checkpoint_root is not None:
        parser.error("use only one of --checkpoint-path and --checkpoint-root")
    write_json(args.output_root / "protocol.json", protocol_manifest())
    prepared_data = build_data(args.dataset, args.seeds[0])
    test_name = PROTOCOLS[args.dataset].batteries[0]
    for seed in args.seeds:
        checkpoint_path = args.checkpoint_path
        if args.checkpoint_root is not None:
            checkpoint_path = (
                args.checkpoint_root / "sg-dits" / args.dataset / test_name
                / f"seed_{seed}" / "checkpoint.pt"
            )
        run(
            args.dataset, seed, args.output_root.resolve(), torch.device(args.device),
            args.max_epochs, args.samples, checkpoint_path, prepared_data,
        )


if __name__ == "__main__":
    main()
