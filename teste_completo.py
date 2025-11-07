# -*- coding: utf-8 -*-
"""
Pipeline completo para segmentação com U-Net (TensorFlow/Keras)

Etapas:
1) Varredura do dataset original, casamento por nome (stem) e filtro de máscaras não-vazias
2) Cópia dos pares válidos para pastas *_clean
3) Conversão de máscaras coloridas (RGB) para IDs de classe (PNG 8-bit)
4) Criação de tf.data (train/val) com resize consistente
5) Definição dos blocos (conv/encoder/decoder) e do modelo U-Net
6) Compilação (Sparse CE from logits) e treinamento com callbacks

Requisitos:
- python 3.9+
- tensorflow 2.10+ (ou compatível com sua GPU)
- opencv-python, numpy
"""

import os
import random
import shutil
from pathlib import Path

import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =========================
# CONFIGURAÇÕES
# =========================
# Raiz do dataset original
DATASET_PATH = "/Users/rodrigogarcia/Documents/cursor projects/teste_vid_01/dataset"
IMAGES_DIR = os.path.join(DATASET_PATH, "images")
MASKS_DIR  = os.path.join(DATASET_PATH, "masks")

# Saídas intermediárias
IMAGES_CLEAN_DIR = "images_clean"
MASKS_CLEAN_DIR  = "masks_clean"
MASKS_CONVERTED_DIR = "masks_converted"

# Mapeamento de cor RGB -> ID (ajuste se suas cores forem outras)
COLOR_TO_ID = {
    (0,   0,   0): 0,  # fundo
    (51,  221, 255): 1,  # classe 1
    (102, 255, 102): 2,  # classe 2
}
# Tolerância em norma L2 para lidar com antialias/compressão (0 = match exato)
TOLERANCE = 0

# Treino
IMG_SIZE = (256, 256)    # altura, largura
BATCH_SIZE = 8
EPOCHS = 20              # aumente conforme necessidade
LR = 1e-3
VAL_SPLIT = 0.2
SEED = 1337

# Data augmentation (leve e segura para segmentação)
USE_AUG = True
FLIP_LR = True
FLIP_UD = False
ROT90 = False  # rotacionar 90° (mantém geometria exata)


# =========================
# UTILITÁRIOS
# =========================
def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def list_pairs(img_dir, msk_dir, img_exts={".jpg", ".jpeg", ".png"}, msk_exts={".png", ".jpg", ".jpeg"}):
    """
    Casa imagens e máscaras por 'stem' (nome sem extensão) e filtra máscaras não-vazias.
    """
    img_dir, msk_dir = Path(img_dir), Path(msk_dir)
    imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in img_exts}
    msks = {p.stem: p for p in msk_dir.iterdir() if p.suffix.lower() in msk_exts}
    pairs = []
    for stem, mpath in msks.items():
        ipath = imgs.get(stem)
        if ipath is None:
            continue
        m = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        if m.ndim == 3:
            m_chk = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        else:
            m_chk = m
        if np.mean(m_chk) > 0:
            pairs.append((str(ipath), str(mpath)))
    return pairs

def copy_pairs(pairs, out_img_dir, out_msk_dir):
    ensure_dir(out_img_dir)
    ensure_dir(out_msk_dir)
    for ipath, mpath in pairs:
        shutil.copy(ipath, os.path.join(out_img_dir, os.path.basename(ipath)))
        shutil.copy(mpath, os.path.join(out_msk_dir, os.path.basename(mpath)))

def convert_masks(input_dir, output_dir, color_to_id, tol=0):
    """
    Converte máscaras coloridas (RGB) para PNGs 8-bit de IDs por pixel.
    """
    ensure_dir(output_dir)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(input_dir, fname)
        m_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if m_bgr is None:
            continue
        m = cv2.cvtColor(m_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = m.shape
        out = np.zeros((h, w), np.uint8)
        for color, idx in color_to_id.items():
            if tol == 0:
                matches = np.all(m == color, axis=-1)
            else:
                # usa int16 para evitar overflow ao subtrair
                matches = np.linalg.norm(m.astype(np.int16) - np.array(color, dtype=np.int16), axis=-1) <= tol
            out[matches] = idx
        out_path = os.path.join(output_dir, Path(fname).stem + ".png")
        cv2.imwrite(out_path, out)
    print(f"✅ Máscaras convertidas em: {output_dir}")

def check_masks_unique(mask_paths, k=5):
    """
    Sanity check rápido de valores únicos em algumas máscaras convertidas.
    """
    print("Sanity check de máscaras convertidas:")
    sample = random.sample(mask_paths, min(k, len(mask_paths)))
    for p in sample:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        uniq = np.unique(m)
        print(f" - {Path(p).name}: unique={uniq}")


# =========================
# DATASET TF.DATA
# =========================
def make_datasets(img_paths, msk_paths, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                  val_split=VAL_SPLIT, seed=SEED):
    assert len(img_paths) == len(msk_paths) and len(img_paths) > 0
    n = len(img_paths)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_split))
    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]

    trn_imgs = tf.constant([img_paths[i] for i in trn_idx])
    trn_msks = tf.constant([msk_paths[i] for i in trn_idx])
    val_imgs = tf.constant([img_paths[i] for i in val_idx])
    val_msks = tf.constant([msk_paths[i] for i in val_idx])

    def load_pair(ipath, mpath):
        img = tf.io.read_file(ipath)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32) / 255.0

        msk = tf.io.read_file(mpath)
        msk = tf.image.decode_png(msk, channels=1)  # IDs 0..C-1
        msk = tf.image.resize(msk, img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        msk = tf.cast(msk, tf.int32)  # (H, W, 1)
        return img, msk

    @tf.function
    def aug(img, msk):
        if FLIP_LR:
            flip = tf.random.uniform(()) > 0.5
            img = tf.cond(flip, lambda: tf.image.flip_left_right(img), lambda: img)
            msk = tf.cond(flip, lambda: tf.image.flip_left_right(msk), lambda: msk)
        if FLIP_UD:
            flip = tf.random.uniform(()) > 0.5
            img = tf.cond(flip, lambda: tf.image.flip_up_down(img), lambda: img)
            msk = tf.cond(flip, lambda: tf.image.flip_up_down(msk), lambda: msk)
        if ROT90:
            k = tf.random.uniform((), maxval=4, dtype=tf.int32)
            img = tf.image.rot90(img, k)
            msk = tf.image.rot90(msk, k)
        return img, msk

    def build_ds(imgs, msks, shuffle, augment):
        ds = tf.data.Dataset.from_tensor_slices((imgs, msks))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(imgs), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = build_ds(trn_imgs, trn_msks, shuffle=True, augment=USE_AUG)
    val_ds   = build_ds(val_imgs, val_msks, shuffle=False, augment=False)
    return train_ds, val_ds


# =========================
# BLOCOS E MODELO U-NET
# =========================
def conv_block(x, filters, kernel_size=3, use_bn=True):
    """Dois Conv2D + (BN) + ReLU."""
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def encoder_block(x, filters, pool=True):
    """Bloco encoder: conv_block + MaxPool (opcional). Retorna (feat, pooled)."""
    c = conv_block(x, filters)
    p = layers.MaxPooling2D()(c) if pool else c
    return c, p

def decoder_block(x, skip, filters):
    """Up-conv (transpose) + concat skip + conv_block."""
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def build_unet(input_shape, n_classes, base_filters=64, use_bn=True):
    """
    U-Net clássica:
    - 4 níveis de downsampling
    - bottleneck
    - 4 níveis de upsampling
    """
    inputs = layers.Input(shape=input_shape)

    c1, p1 = encoder_block(inputs, base_filters, pool=True)            # 64
    c2, p2 = encoder_block(p1, base_filters*2, pool=True)             # 128
    c3, p3 = encoder_block(p2, base_filters*4, pool=True)             # 256
    c4, p4 = encoder_block(p3, base_filters*8, pool=True)             # 512

    bn = conv_block(p4, base_filters*16)                               # 1024

    d1 = decoder_block(bn, c4, base_filters*8)                         # 512
    d2 = decoder_block(d1, c3, base_filters*4)                         # 256
    d3 = decoder_block(d2, c2, base_filters*2)                         # 128
    d4 = decoder_block(d3, c1, base_filters)                           # 64

    # Logits por classe (sem softmax; usaremos from_logits=True)
    outputs = layers.Conv2D(n_classes, 1, activation=None)(d4)

    model = keras.Model(inputs, outputs, name="unet")
    return model

class SparseMeanIoU(tf.keras.metrics.MeanIoU):
    """MeanIoU para rótulos esparsos e y_pred com logits/probabilidades (B,H,W,C)."""
    def __init__(self, num_classes, name="miou", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true: (B,H,W,1) com IDs; y_pred: (B,H,W,C) logits/probs
        y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)  # (B,H,W)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)  # (B,H,W)
        return super().update_state(y_true, y_pred, sample_weight)


# =========================
# MAIN (treino)
# =========================
def main():
    set_seeds(SEED)

    # 1) Lista pares não-vazios no dataset original
    pairs = list_pairs(IMAGES_DIR, MASKS_DIR)
    if len(pairs) == 0:
        raise RuntimeError("Nenhum par imagem-máscara encontrado ou todas as máscaras estão vazias.")
    print(f"✔ Pares válidos detectados: {len(pairs)}")

    # 2) Copia pares p/ pastas clean (não apaga conteúdo antigo automaticamente)
    ensure_dir(IMAGES_CLEAN_DIR); ensure_dir(MASKS_CLEAN_DIR); ensure_dir(MASKS_CONVERTED_DIR)
    copy_pairs(pairs, IMAGES_CLEAN_DIR, MASKS_CLEAN_DIR)
    print(f"✔ Cópia concluída -> {IMAGES_CLEAN_DIR} / {MASKS_CLEAN_DIR}")

    # 3) Converte máscaras coloridas -> IDs
    convert_masks(MASKS_CLEAN_DIR, MASKS_CONVERTED_DIR, COLOR_TO_ID, tol=TOLERANCE)

    # 4) Gera lista final de pares (imagem_clean, máscara_convertida)
    clean_pairs = list_pairs(IMAGES_CLEAN_DIR, MASKS_CONVERTED_DIR)
    if len(clean_pairs) == 0:
        raise RuntimeError("Após a conversão, não há pares válidos. Verifique COLOR_TO_ID/TOLERANCE.")
    img_paths = [p[0] for p in clean_pairs]
    msk_paths = [p[1] for p in clean_pairs]

    # Sanity check
    check_masks_unique(msk_paths, k=5)

    # 5) tf.data
    train_ds, val_ds = make_datasets(img_paths, msk_paths)

    # 6) Modelo (blocos + U-Net)
    n_classes = max(COLOR_TO_ID.values()) + 1
    model = build_unet(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), n_classes=n_classes, base_filters=64)
    model.summary()

    # 7) Compile
    n_classes = max(COLOR_TO_ID.values()) + 1
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = keras.optimizers.Adam(LR)
    metrics = [
    keras.metrics.SparseCategoricalAccuracy(name="acc"),
    SparseMeanIoU(num_classes=n_classes, name="miou"),
    ]
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    # 8) Callbacks
    cbs = [
        keras.callbacks.ModelCheckpoint(
            "unet_best.keras", monitor="val_miou", mode="max",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.CSVLogger("training_log.csv", append=False),
    ]

    # 9) Treino
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cbs
    )

    # 10) Inferência rápida num batch de validação
    for batch in val_ds.take(1):
        imgs, msks = batch
        logits = model(imgs, training=False)
        preds = tf.argmax(logits, axis=-1)  # (B, H, W)
        unique_vals = tf.sort(tf.unique(tf.reshape(preds, [-1]))[0]).numpy()
        print(f"Pred labels únicos (amostra val): {unique_vals}")
        break

    print("✔ Fim. Melhor modelo salvo em: unet_best.keras | Log: training_log.csv")


if __name__ == "__main__":
    # Evita alocação total da GPU (se houver)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("Aviso (GPU):", e)
    main()
