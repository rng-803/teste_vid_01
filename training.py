# -*- coding: utf-8 -*-
"""
Funções relacionadas ao treinamento e métricas.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def batch_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return (preds == targets).float().mean().item()


def batch_mean_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    ious: List[float] = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    steps = 0

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            logits = model(imgs)
            loss = criterion(logits, masks)
            preds = torch.argmax(logits, dim=1)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        total_acc += batch_accuracy(preds, masks)
        total_iou += batch_mean_iou(preds, masks, num_classes)
        steps += 1

    if steps == 0:
        return 0.0, 0.0, 0.0
    return total_loss / steps, total_acc / steps, total_iou / steps


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    epochs: int,
    best_model_path: str,
) -> Tuple[List[Tuple[int | str, str, float, float, float]], float]:
    history_rows: List[Tuple[int | str, str, float, float, float]] = [("epoch", "split", "loss", "acc", "miou")]
    best_val_iou = -1.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_iou = run_epoch(
            model, train_loader, criterion, optimizer, device, num_classes
        )
        val_loss, val_acc, val_iou = run_epoch(
            model, val_loader, criterion, None, device, num_classes
        )
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} miou={train_iou:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} miou={val_iou:.4f}"
        )
        history_rows.append((epoch, "train", train_loss, train_acc, train_iou))
        history_rows.append((epoch, "val", val_loss, val_acc, val_iou))
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Novo melhor modelo salvo em {best_model_path} (val IoU={val_iou:.4f})")

    return history_rows, best_val_iou


def preview_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Retorna os rótulos previstos do primeiro batch de validação."""
    model.eval()
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            preds = torch.argmax(model(imgs), dim=1)
            return torch.unique(preds).cpu().numpy()
    return np.array([])
