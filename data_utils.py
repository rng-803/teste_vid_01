# -*- coding: utf-8 -*-
"""
Funções utilitárias para preparação de dados.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def list_pairs(
    img_dir: str | Path,
    msk_dir: str | Path,
    img_exts: Sequence[str] = (".jpg", ".jpeg", ".png"),
    msk_exts: Sequence[str] = (".png", ".jpg", ".jpeg"),
) -> List[Tuple[str, str]]:
    """Casa imagens e máscaras pelo stem e descarta máscaras vazias."""
    img_dir = Path(img_dir)
    msk_dir = Path(msk_dir)
    imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in img_exts}
    msks = {p.stem: p for p in msk_dir.iterdir() if p.suffix.lower() in msk_exts}
    pairs: List[Tuple[str, str]] = []
    for stem, mpath in msks.items():
        ipath = imgs.get(stem)
        if ipath is None:
            continue
        mask = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.ndim == 3 else mask
        if np.mean(mask_gray) > 0:
            pairs.append((str(ipath), str(mpath)))
    return pairs


def copy_pairs(pairs: Sequence[Tuple[str, str]], out_img_dir: str, out_msk_dir: str) -> None:
    ensure_dir(out_img_dir)
    ensure_dir(out_msk_dir)
    for ipath, mpath in pairs:
        shutil.copy(ipath, os.path.join(out_img_dir, os.path.basename(ipath)))
        shutil.copy(mpath, os.path.join(out_msk_dir, os.path.basename(mpath)))


def convert_masks(
    input_dir: str,
    output_dir: str,
    color_to_id: dict,
    tol: int = 0,
) -> None:
    """Converte máscaras RGB para PNG 8-bit com IDs por pixel."""
    ensure_dir(output_dir)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(input_dir, fname)
        mask_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if mask_bgr is None:
            continue
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = mask_rgb.shape
        out = np.zeros((h, w), np.uint8)
        for color, idx in color_to_id.items():
            if tol == 0:
                matches = np.all(mask_rgb == color, axis=-1)
            else:
                matches = (
                    np.linalg.norm(
                        mask_rgb.astype(np.int16) - np.array(color, dtype=np.int16), axis=-1
                    )
                    <= tol
                )
            out[matches] = idx
        out_path = os.path.join(output_dir, Path(fname).stem + ".png")
        cv2.imwrite(out_path, out)
    print(f"✅ Máscaras convertidas em: {output_dir}")


def check_masks_unique(mask_paths: Sequence[str], k: int = 5) -> None:
    """Mostra valores únicos em algumas máscaras convertidas."""
    print("Sanity check de máscaras convertidas:")
    sample = random.sample(mask_paths, min(k, len(mask_paths)))
    for p in sample:
        mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        uniq = np.unique(mask)
        print(f" - {Path(p).name}: unique={uniq}")
