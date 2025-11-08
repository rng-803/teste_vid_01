# -*- coding: utf-8 -*-
"""
Wrapper simples em Gradio para visualizar inferências da U-Net:
- imagem original
- máscara ground-truth convertida em cores
- máscara prevista pelo modelo treinado
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import gradio as gr
import numpy as np
import torch

from config import (
    BEST_MODEL_PATH,
    COLOR_TO_ID,
    IMG_SIZE,
    IMAGES_CLEAN_DIR,
    MASKS_CONVERTED_DIR,
)
from data_utils import list_pairs
from model_unet import UNet

ID_TO_COLOR: Dict[int, Tuple[int, int, int]] = {idx: color for color, idx in COLOR_TO_ID.items()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = max(COLOR_TO_ID.values()) + 1


def load_model() -> UNet:
    model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_filters=64)
    state = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(img_path: Path) -> Tuple[torch.Tensor, Tuple[int, int]]:
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Falha ao carregar imagem: {img_path}")
    original_shape = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_LINEAR)
    img_resized = img_resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).unsqueeze(0)
    return tensor, original_shape


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    color_mask = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for idx, rgb in ID_TO_COLOR.items():
        color_mask[mask == idx] = rgb
    return color_mask


def resize_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def resolve_image_path(image_stem: str) -> Path:
    base_dir = Path(IMAGES_CLEAN_DIR)
    candidates = list(base_dir.glob(f"{image_stem}.*"))
    if not candidates:
        raise FileNotFoundError(f"Imagem não encontrada para {image_stem}")
    return candidates[0]


def run_inference(model: UNet, image_stem: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_path = resolve_image_path(image_stem)
    mask_path = Path(MASKS_CONVERTED_DIR) / f"{image_stem}.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"Máscara GT não encontrada para {image_stem}")

    original_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise RuntimeError(f"Falha ao carregar {img_path}")
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        raise RuntimeError(f"Falha ao carregar {mask_path}")
    gt_color = mask_to_color(gt_mask)

    tensor, orig_shape = preprocess_image(img_path)
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    pred = resize_mask(pred, orig_shape)
    pred_color = mask_to_color(pred)
    return original_rgb, gt_color, pred_color


def build_interface(model: UNet, stems: List[str]) -> gr.Blocks:
    with gr.Blocks(title="Preview Inferência U-Net") as demo:
        gr.Markdown("# Preview de Inferência (U-Net PyTorch)")
        dropdown = gr.Dropdown(choices=stems, value=stems[0], label="Selecione a imagem")
        with gr.Row():
            orig_out = gr.Image(label="Imagem Original", type="numpy")
            gt_out = gr.Image(label="Máscara GT", type="numpy")
            pred_out = gr.Image(label="Predição", type="numpy")

        def _update(stem: str):
            return run_inference(model, stem)

        dropdown.change(_update, inputs=dropdown, outputs=[orig_out, gt_out, pred_out])
        gr.Button("Atualizar").click(_update, inputs=dropdown, outputs=[orig_out, gt_out, pred_out])
        gr.HTML(
            "Certifique-se de ter treinado o modelo e salvo os pesos em "
            f"<code>{BEST_MODEL_PATH}</code> antes de iniciar o preview."
        )
        demo.load(_update, inputs=dropdown, outputs=[orig_out, gt_out, pred_out])
    return demo


def main() -> None:
    pairs = list_pairs(IMAGES_CLEAN_DIR, MASKS_CONVERTED_DIR)
    if not pairs:
        raise RuntimeError("Nenhuma imagem/máscara encontrada nas pastas *_clean.")
    stems = sorted({Path(img).stem for img, _ in pairs})
    model = load_model()
    demo = build_interface(model, stems)
    demo.launch()


if __name__ == "__main__":
    main()
