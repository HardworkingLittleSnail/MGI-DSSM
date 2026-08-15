from __future__ import annotations

import random
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import ExperimentConfig
from .model import BATTERMoE


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict(model: BATTERMoE, dataset: Dataset, batch_size: int, device: torch.device):
    model.eval()
    predictions, targets = [], []
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        predictions.append(model(x.to(device)).prediction.cpu())
        targets.append(y)
    return torch.cat(predictions).numpy(), torch.cat(targets).numpy()


@torch.no_grad()
def validation_mae(model: BATTERMoE, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    absolute, count = 0.0, 0
    for x, y in loader:
        y = y.to(device)
        prediction = model(x.to(device)).prediction
        absolute += nn.functional.l1_loss(prediction, y, reduction='sum').item()
        count += len(y)
    return absolute / max(count, 1)


def fit(
    model: BATTERMoE,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    config: ExperimentConfig,
    device: torch.device,
    seed: int = 42,
    verbose_prefix: str | None = None,
    minimum_checkpoint_epoch: int = 0,
) -> tuple[BATTERMoE, list[dict[str, float]]]:
    seed_everything(seed)
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = None
    if 0.0 < config.lr_plateau_factor < 1.0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.lr_plateau_factor,
            patience=config.lr_plateau_patience,
            min_lr=config.minimum_learning_rate,
        )
    val_mae = validation_mae(model, val_loader, device)
    best_state = deepcopy(model.state_dict()) if minimum_checkpoint_epoch == 0 else None
    best_mae = val_mae if minimum_checkpoint_epoch == 0 else float('inf')
    stale = 0
    history = [{
        'epoch': 0,
        'train_loss': float('nan'),
        'validation_mae': val_mae,
        'learning_rate': optimizer.param_groups[0]['lr'],
    }]
    if verbose_prefix:
        print(
            f"{verbose_prefix} epoch=000 train=nan val_mae={val_mae:.7f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total, samples = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            loss, _ = model.loss(output, y)
            loss.backward()
            if config.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            total += loss.item() * len(y)
            samples += len(y)
        val_mae = validation_mae(model, val_loader, device)
        if scheduler is not None:
            scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]['lr']
        history.append({
            'epoch': epoch,
            'train_loss': total / samples,
            'validation_mae': val_mae,
            'learning_rate': current_lr,
        })
        if verbose_prefix:
            print(
                f"{verbose_prefix} epoch={epoch:03d} "
                f"train={history[-1]['train_loss']:.7f} val_mae={val_mae:.7f} "
                f"lr={current_lr:.2e}",
                flush=True,
            )
        if epoch >= minimum_checkpoint_epoch and val_mae < best_mae:
            best_mae, stale = val_mae, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError('No eligible checkpoint was produced during training')
    model.load_state_dict(best_state)
    return model, history
