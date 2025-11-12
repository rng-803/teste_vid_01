# -*- coding: utf-8 -*-
"""
Pipeline completo para segmentação com U-Net em PyTorch.

O script principal agora delega tarefas para módulos menores:
- config.py: constantes e hiperparâmetros
- data_utils.py: preparo do dataset e conversão de máscaras
- datasets.py: Dataset/DataLoader com augmentations
- model_unet.py: definição da rede U-Net
- training.py: laço de treino, métricas e preview de previsões
"""

from __future__ import annotations

import csv
import torch
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torch.amp import GradScaler

from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    COLOR_TO_ID,
    EPOCHS,
    FLIP_LR,
    FLIP_UD,
    IMG_SIZE,
    IMAGES_CLEAN_DIR,
    IMAGES_DIR,
    LR,
    MASKS_CLEAN_DIR,
    MASKS_CONVERTED_DIR,
    MASKS_DIR,
    ROT90,
    SEED,
    TOLERANCE,
    TRAIN_LOG,
    USE_AUG,
    VAL_SPLIT,
)
from data_utils import (
    check_masks_unique,
    convert_masks,
    copy_pairs,
    ensure_dir,
    list_pairs,
    set_seeds,
)
from datasets import make_dataloaders
from model_unet import UNet
from training import preview_predictions, train_model, dice_loss


def main() -> None:
    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    pairs = list_pairs(IMAGES_DIR, MASKS_DIR)
    if not pairs:
        raise RuntimeError("Nenhum par imagem-máscara encontrado.")
    print(f"✔ Pares válidos detectados: {len(pairs)}")

    ensure_dir(IMAGES_CLEAN_DIR)
    ensure_dir(MASKS_CLEAN_DIR)
    ensure_dir(MASKS_CONVERTED_DIR)
    copy_pairs(pairs, IMAGES_CLEAN_DIR, MASKS_CLEAN_DIR)
    print(f"✔ Cópia concluída -> {IMAGES_CLEAN_DIR} / {MASKS_CLEAN_DIR}")

    convert_masks(MASKS_CLEAN_DIR, MASKS_CONVERTED_DIR, COLOR_TO_ID, tol=TOLERANCE)

    clean_pairs = list_pairs(IMAGES_CLEAN_DIR, MASKS_CONVERTED_DIR)
    if not clean_pairs:
        raise RuntimeError("Após a conversão, não há pares válidos. Ajuste COLOR_TO_ID/TOLERANCE.")
    img_paths = [p[0] for p in clean_pairs]
    mask_paths = [p[1] for p in clean_pairs]

    check_masks_unique(mask_paths, k=5)

    train_loader, val_loader = make_dataloaders(
        img_paths,
        mask_paths,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        seed=SEED,
        use_aug=USE_AUG,
        flip_lr=FLIP_LR,
        flip_ud=FLIP_UD,
        rot90=ROT90,
    )

    num_classes = max(COLOR_TO_ID.values()) + 1

    model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes,
).to(device)

    # Optional: add class weights if you want to handle imbalance
    class_weights = torch.tensor([1.0, 2.0, 0.5], device=device)

    ce = nn.CrossEntropyLoss(weight=class_weights)
    def criterion(logits, y):
        return 0.5 * ce(logits, y) + 0.5 * dice_loss(logits, y, num_classes=num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())

    history_rows = [("epoch", "split", "loss", "acc", "miou")]
    best_val_iou = -1.0
    patience = 10
    stale = 0


    history_rows, _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_classes=num_classes,
        epochs=EPOCHS,
    )

    with open(TRAIN_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(history_rows)
    print(f"✔ Treino concluído. Log salvo em {TRAIN_LOG}")

    unique_vals = preview_predictions(model, val_loader, device)
    print(f"Pred labels únicos (amostra val): {unique_vals}")


if __name__ == "__main__":
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception as err:  # pragma: no cover
        print("Aviso (CUDNN):", err)
    main()

