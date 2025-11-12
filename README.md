# UNET a partir de modelo RESNET pré treinado

Pipeline de segmentação semântica em **vídeos cirúrgicos** com **3 classes**, usando PyTorch + U‑Net (via `segmentation-models-pytorch`). O repositório inclui:

- preparo automatizado das pastas `images_clean` / `masks_clean` / `masks_converted`;
- treinamento completo (`teste_completo_torch.py`) com augmentations via Albumentations e mixed precision (GradScaler);
- avaliação de métricas clássicas (mIoU, precision, recall) em `evaluate_torch.py`;
- scripts de inferência (`inference_torch.py`) e visualização interativa (Gradio) em `gradio_preview.py`.

> Dataset base (frames + máscaras) hospedado em: https://www.kaggle.com/datasets/rngarcia/vid-01-incomplete

---

## ✨ Principais recursos
- U-Net com backbone `resnet34` pré-treinado (ImageNet) via `segmentation-models-pytorch`.
- Data augmentation com Albumentations (resize, flips, rotações, jitter, blur).
- Treino em mixed precision (autocast + GradScaler) e scheduler ReduceLROnPlateau.
- Perda híbrida (CrossEntropy ponderada + Dice) para lidar com desbalanceamento.
- Métricas de avaliação dedicadas (mIoU, precisão, recall) em script separado.
- Interface Gradio para inspecionar visualmente imagens, GT e predição, com opção de download.

---

## 🗂️ Estrutura atual do projeto

```
teste_vid_01/
├─ dataset/                 # dataset original (images/masks)
├─ images_clean/            # imagens válidas copiadas
├─ masks_clean/             # máscaras originais correspondentes
├─ masks_converted/         # máscaras 8-bit com IDs
├─ config.py                # constantes e hiperparâmetros
├─ data_utils.py            # funções de varredura, cópia e conversão
├─ datasets.py              # Dataset/DataLoader + Albumentations
├─ model_unet.py            # definição da U-Net (caso queira custom)
├─ training.py              # laço de treino, métricas, GradScaler
├─ teste_completo_torch.py  # orquestração do pipeline de treino
├─ inference_torch.py       # inferência batch em um diretório
├─ evaluate_torch.py        # cálculo de métricas no conjunto limpo
├─ gradio_preview.py        # visualização interativa (Gradio)
├─ requirements.txt
└─ README.md
```

> Os diretórios `images_clean/`, `masks_clean/` e `masks_converted/` são gerados automaticamente ao rodar `teste_completo_torch.py`. O dataset original permanece em `dataset/images` e `dataset/masks`.

---

## 📦 Instalação

Pré-requisitos: Python 3.9+ (GPU opcional, mas recomendado).

```bash
# 1) Ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1

# 2) Dependências
pip install --upgrade pip
pip install -r requirements.txt
```

> Para CUDA específica, consulte https://pytorch.org/get-started/locally/ e ajuste o comando de instalação do PyTorch antes de `pip install -r requirements.txt`.

---

## 🗃️ Dataset

Baixe do Kaggle:
```bash
pip install kaggle
kaggle datasets download -d rngarcia/vid-01-incomplete
unzip vid-01-incomplete.zip -d dataset/
```
Estrutura esperada:
```
dataset/
 ├─ images/   # frames RGB
 └─ masks/    # máscaras (mesma nomenclatura dos frames)
```

O script principal cuidará de criar versões “limpas” e converter máscaras RGB para IDs usando `COLOR_TO_ID` (ajuste em `config.py` se suas cores/classes mudarem).

---

## 🚀 Fluxo principal

### 1. Treinamento
```bash
source .venv/bin/activate
python teste_completo_torch.py
```
Isso executa:
1. Casamento imagem/máscara original e filtro de máscaras vazias.
2. Cópia para pastas clean + conversão para ids (PNG 8-bit).
3. Split train/val (`VAL_SPLIT`), criação de DataLoaders com Albumentations.
4. Treino da U-Net (ResNet34 encoder) com CE + Dice, GradScaler e early stop.
5. Salvamento do melhor modelo em `unet_best_torch.pt` e log em `training_log_torch.csv`.

### 2. Avaliação
```bash
python evaluate_torch.py --weights unet_best_torch.pt
```
Exibe mIoU, precisão média e recall médio sobre todo o conjunto limpo. Ajuste `--batch-size`, `--images-dir`, `--masks-dir` conforme necessário.

### 3. Inferência batch
```bash
python inference_torch.py \
  --images_dir images_clean \
  --output_dir predictions \
  --weights unet_best_torch.pt \
  --color
```
Gera máscaras previstas (e opcionalmente coloridas) para cada imagem.

### 4. Preview interativo (Gradio)
```bash
python gradio_preview.py
```
Abre uma UI local onde é possível escolher uma imagem, inspecionar Original × GT × Predição e baixar o PNG da predição.

---

## ⚙️ Configurações / hiperparâmetros

Edite `config.py` para ajustar:
- Caminhos base do dataset (`DATASET_PATH`, `IMAGES_DIR`, `MASKS_DIR`).
- Mapeamento RGB→ID (`COLOR_TO_ID`).
- `IMG_SIZE`, `BATCH_SIZE`, `EPOCHS`, `LR`, `VAL_SPLIT`, `SEED`.
- Flags de augmentation (`USE_AUG`, `FLIP_LR`, etc.) – Albumentations também pode ser ajustado em `datasets.py` (`get_train_transforms`).

Pesos de classe da CrossEntropy estão definidos diretamente em `teste_completo_torch.py`; ajuste o tensor `class_weights` conforme sua distribuição.

---

## 📊 Métricas

- **Durante o treino**: accuracy e IoU médios por epoch (console + CSV).
- **Avaliação dedicada**: `evaluate_torch.py` calcula matriz de confusão e deriva mIoU, precisão macro e recall macro.
- **Preview**: `gradio_preview.py` mostra qualitativamente os resultados.

---

## 🔧 Dicas adicionais

- Garanta que os diretórios de dados (`dataset/`, `images_clean/`, etc.) estejam no `.gitignore` caso não queira versioná-los.
- Para experimentar outros backbones (ex.: `efficientnet-b3`), basta ajustar `encoder_name` no construtor `smp.Unet` e instalar o encoder correspondente (já coberto por `timm`).
- Se quiser rodar em CPU, o código funciona, porém o treinamento será mais lento e o GradScaler será automaticamente desativado.

---

## ✔️ Requirements atuais

```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.23.0
opencv-python>=4.8.0
albumentations>=1.3.0
segmentation-models-pytorch>=0.3.3
timm>=0.9.2
gradio>=4.0.0
```

Instale-os via `pip install -r requirements.txt` (após configurar PyTorch adequado para seu hardware, se necessário).

---

## 📝 Licença / contato

Adapte esta seção conforme sua necessidade (ex.: MIT, CC BY-NC, etc.).

