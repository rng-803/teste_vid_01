# Teste vid_01

Segmentação semântica/instância em **vídeos cirúrgicos** com **3 classes** usando **U‑Net em PyTorch**.  
Objetivo: **treinar** o modelo e disponibilizar **script de inferência**.

> Dataset hospedado no Kaggle: https://www.kaggle.com/datasets/rngarcia/vid-01-incomplete

---

## ✨ Principais recursos
- Pipeline de **treinamento** em PyTorch (U‑Net).
- **Inferência** em frames ou vídeo completo.
- Métricas clássicas (IoU/Dice) e logs de treinamento.
- Scripts para **download e preparação de dados** (via Kaggle API).
- Estrutura clara de projeto e reprodutibilidade.

---

## 🗂️ Estrutura recomendada do projeto

```
teste-vid_01/
├─ data/
│  ├─ images/           # frames .png/.jpg
│  ├─ masks/            # máscaras com 3 classes (1 canal)
│  └─ metadata.csv      # opcional
├─ src/
│  ├─ models/
│  │  └─ unet.py
│  ├─ data/
│  │  ├─ dataset.py
│  │  └─ transforms.py
│  ├─ train.py
│  ├─ infer.py
│  └─ utils.py
├─ notebooks/           # opcional, EDA/visualizações
├─ README.md
├─ requirements.txt
└─ .gitignore
```

> Observação: adapte os nomes dos arquivos se sua estrutura já estiver diferente.  
> Em visão computacional, é comum manter **`data/` fora do Git** e apenas documentar como obtê-lo.

---

## 📦 Instalação

Requer: Python 3.9+ (recomendado), CUDA opcional.

```bash
# 1) Crie e ative um ambiente virtual (exemplos)
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\\Scripts\\Activate.ps1

# 2) Instale dependências
pip install --upgrade pip
pip install -r requirements.txt
```

> O projeto foi atualizado para **PyTorch**. Este README assume o uso de GPU, mas funciona em CPU (mais lento).  
> Para CUDA específica, consulte a tabela de compatibilidade no site do PyTorch e ajuste a instalação se necessário.

---

## 🗃️ Dataset (Kaggle)

Link público:
```
https://www.kaggle.com/datasets/rngarcia/vid-01-incomplete
```

### Baixar via API do Kaggle

1) Configure suas credenciais em `~/.kaggle/kaggle.json` (consulte sua conta Kaggle → *Create New API Token*).  
2) Execute:

```bash
pip install kaggle
kaggle datasets download -d rngarcia/vid-01-incomplete
unzip vid-01-incomplete.zip -d data/
```

### Estrutura esperada após extração
```
data/
 ├─ images/   # frames
 ├─ masks/    # máscaras 1-canal com valores de classe (0..3, por ex.)
 └─ metadata.csv (opcional)
```

> **Classes (3 no total):** ajuste os valores/cores conforme o padrão das suas máscaras.  
> Se usar paleta/cores diferentes, atualize o `dataset.py` para ler corretamente.

---

## 🚀 Uso rápido

### Treinamento
Exemplo de execução (hipotético; ajuste flags conforme `src/train.py`):

```bash
python -m src.train \
  --data_dir data \
  --images_dir images \
  --masks_dir masks \
  --num_classes 3 \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-3 \
  --out_dir runs/exp01
```

### Inferência em vídeo ou frames
```bash
python -m src.infer \
  --weights runs/exp01/best.pt \
  --input path/to/video.mp4 \
  --output runs/exp01/preds.mp4 \
  --num_classes 3
```

> Para frames, passe `--input data/images` e `--output runs/exp01/preds/`

---

## 🧠 Modelo: U‑Net (PyTorch)

- Implementação em `src/models/unet.py` (encoder–decoder com skip connections).
- Recomenda-se usar **`num_classes=3`** (canais de saída = nº de classes).
- Funções de perda comuns: **Dice Loss**, **Cross‑Entropy**, **Focal** (ajuste conforme seu desequilíbrio de classes).
- Métricas: **IoU** e **Dice** por classe e média.

> Máscaras: use **1 canal** com rótulos inteiros (0..C‑1). Para PNG indexado, garanta que a leitura preserve índices.

---

## ⚙️ Configurações e reprodutibilidade

- Fixe seeds (PyTorch, NumPy) no `train.py` para runs comparáveis.
- Logue hiperparâmetros e métricas (ex.: CSV simples, TensorBoard, Weights & Biases – opcional).
- Salve `best.pt` por melhor IoU/Dice de validação.

---

## 🧪 Validação / Métricas

- **Split sugerido:** 70/15/15 (train/val/test) estratificado por vídeo/caso.
- Reporte **IoU/Dice por classe** e médias (mIoU, mDice).
- Se for vídeo, considere avaliação **temporal** (consistência entre frames).


---



## 🧰 Dicas de `.gitignore` (opcional)

```
# Dados e saídas
data/
runs/
checkpoints/
*.pt
*.pth
*.ckpt

# Ambientes/OS
.venv/
__pycache__/
.DS_Store
```

---
