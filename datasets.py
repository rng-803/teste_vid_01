# -*- coding: utf-8 -*-
"""
Datasets e DataLoaders para o pipeline.
"""

from __future__ import annotations

import random
from typing import Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def resize_img_mask(img: np.ndarray, mask: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = size
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return img, mask


def apply_aug(
    img: np.ndarray,
    mask: np.ndarray,
    flip_lr: bool,
    flip_ud: bool,
    rot90: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if flip_lr and random.random() < 0.5:
        img = np.fliplr(img)
        mask = np.fliplr(mask)
    if flip_ud and random.random() < 0.5:
        img = np.flipud(img)
        mask = np.flipud(mask)
    if rot90:
        k = random.randint(0, 3)
        img = np.rot90(img, k)
        mask = np.rot90(mask, k)
    return img.copy(), mask.copy()


class SegmentationDataset(Dataset):
    def __init__(
        self,
        img_paths: Sequence[str],
        mask_paths: Sequence[str],
        img_size: Tuple[int, int],
        augment: bool,
        flip_lr: bool,
        flip_ud: bool,
        rot90: bool,
    ):
        assert len(img_paths) == len(mask_paths)
        self.img_paths = list(img_paths)
        self.mask_paths = list(mask_paths)
        self.img_size = img_size
        self.augment = augment
        self.flip_lr = flip_lr
        self.flip_ud = flip_ud
        self.rot90 = rot90

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise RuntimeError(f"Falha ao carregar par: {self.img_paths[idx]} / {self.mask_paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, mask = resize_img_mask(img, mask, self.img_size)
        if self.augment:
            img, mask = apply_aug(img, mask, self.flip_lr, self.flip_ud, self.rot90)
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.int64)
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        mask = torch.from_numpy(mask)
        return img, mask


def make_dataloaders(
    img_paths: Sequence[str],
    mask_paths: Sequence[str],
    img_size: Tuple[int, int],
    batch_size: int,
    val_split: float,
    seed: int,
    use_aug: bool,
    flip_lr: bool,
    flip_ud: bool,
    rot90: bool,
) -> Tuple[DataLoader, DataLoader]:
    assert len(img_paths) == len(mask_paths) and len(img_paths) > 0
    indices = np.arange(len(img_paths))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_split))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_imgs = [img_paths[i] for i in train_idx]
    train_msks = [mask_paths[i] for i in train_idx]
    val_imgs = [img_paths[i] for i in val_idx]
    val_msks = [mask_paths[i] for i in val_idx]

    train_ds = SegmentationDataset(train_imgs, train_msks, img_size, use_aug, flip_lr, flip_ud, rot90)
    val_ds = SegmentationDataset(val_imgs, val_msks, img_size, False, flip_lr, flip_ud, rot90)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader
