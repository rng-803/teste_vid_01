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
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from config import EPOCHS
from config import BEST_MODEL_PATH

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

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice (multi-class). logits: (B,C,H,W), targets: (B,H,W) with int labels.
    """
    probs = torch.softmax(logits, dim=1)                   # (B,C,H,W)
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes)  # (B,H,W,C)
    targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()         # (B,C,H,W)

    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets_onehot, dims)
    union = torch.sum(probs + targets_onehot, dims)
    dice_per_class = (2.0 * intersection + eps) / (union + eps)         # (C,)
    # average over classes present
    return 1.0 - dice_per_class.mean()

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion_fn,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    num_classes: int,
    scaler: GradScaler | None = None,
) -> Tuple[float, float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_iou = 0.0
    steps = 0
    amp_enabled = scaler is not None and device.type == "cuda"

    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with autocast(enabled=amp_enabled):
                logits = model(imgs)
                loss = criterion_fn(logits, masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item()
        total_acc += batch_accuracy(preds, masks)
        total_iou += batch_mean_iou(preds, masks, num_classes)
        steps += 1

    if steps == 0:
        return 0.0, 0.0, 0.0
    return total_loss / steps, total_acc / steps, total_iou / steps


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_classes,
    epochs: int = 20,
    patience: int = 10,
):
    # cria o scaler aqui (não depende de variável externa)
    scaler = GradScaler('cuda', enabled=(device.type == "cuda"))

    # se usa ReduceLROnPlateau, crie aqui também (sem verbose)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history_rows = [("epoch", "split", "loss", "acc", "miou")]
    best_val_iou = -1.0
    stale = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_iou = run_epoch(
            model, train_loader, criterion, optimizer, device, num_classes, scaler
        )
        val_loss, val_acc, val_iou = run_epoch(
            model, val_loader, criterion, None, device, num_classes, None
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} miou={train_iou:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} miou={val_iou:.4f}"
        )

        history_rows.append((epoch, "train", train_loss, train_acc, train_iou))
        history_rows.append((epoch, "val", val_loss, val_acc, val_iou))

        scheduler.step(val_loss)
        # print LR opcional:
        # print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            stale = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"✅ Novo melhor modelo salvo em {BEST_MODEL_PATH} (val IoU={val_iou:.4f})")
        else:
            stale += 1
            if stale >= patience:
                print(f"⏹️ Early stopping (sem melhora por {patience} epochs).")
                break

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

