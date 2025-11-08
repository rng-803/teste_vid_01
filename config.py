# -*- coding: utf-8 -*-
"""
Configurações compartilhadas para o pipeline PyTorch.
"""

DATASET_PATH = "/Users/rodrigogarcia/Documents/cursor projects/teste_vid_01/dataset"
IMAGES_DIR = f"{DATASET_PATH}/images"
MASKS_DIR = f"{DATASET_PATH}/masks"

IMAGES_CLEAN_DIR = "images_clean"
MASKS_CLEAN_DIR = "masks_clean"
MASKS_CONVERTED_DIR = "masks_converted"

COLOR_TO_ID = {
    (0, 0, 0): 0,          # fundo
    (51, 221, 255): 1,     # classe 1
    (102, 255, 102): 2,    # classe 2
}
TOLERANCE = 0

IMG_SIZE = (256, 256)    # (H, W)
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-3
VAL_SPLIT = 0.2
SEED = 1337

USE_AUG = True
FLIP_LR = True
FLIP_UD = False
ROT90 = False

TRAIN_LOG = "training_log_torch.csv"
BEST_MODEL_PATH = "unet_best_torch.pt"
