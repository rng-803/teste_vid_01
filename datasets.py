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
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
    def __init__(self, img_paths: Sequence[str], mask_paths: Sequence[str], img_size: Tuple[int, int], augment: bool):
        assert len(img_paths) == len(mask_paths)
        self.img_paths = list(img_paths)
        self.mask_paths = list(mask_paths)
        self.img_size = img_size
        self.augment = augment
        self.tform = get_train_transforms(img_size) if augment else get_val_transforms(img_size)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise RuntimeError(f"Falha ao carregar par: {self.img_paths[idx]} / {self.mask_paths[idx]}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Albumentations expects HWC image + 2D mask
        aug = self.tform(image=img, mask=mask)
        img, mask = aug["image"], aug["mask"]

        # to tensor (CHW float32) and long mask
        img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        mask = mask.astype(np.int64)

        return torch.from_numpy(img), torch.from_numpy(mask)



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

    train_ds = SegmentationDataset(train_imgs, train_msks, img_size, augment=use_aug)
    val_ds   = SegmentationDataset(val_imgs,   val_msks,   img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def get_train_transforms(size):
    h, w = size
    return A.Compose([
        A.Resize(h, w, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.0),
        A.RandomRotate90(p=0.25),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(blur_limit=(3,5), p=0.2),
    ])

def get_val_transforms(size):
    h, w = size
    return A.Compose([
        A.Resize(h, w, interpolation=cv2.INTER_LINEAR),
    ])
