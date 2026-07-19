# ASP-SNN: Active Spiking Perception for 3D Point Cloud Understanding

A spiking neural network framework that treats 3D point cloud perception as
an iterative decision process — selecting which local patch ("slice") to examine
next based on accumulated spiking neuron dynamics (LIF neurons), rather than
processing the entire point cloud at once.

Evaluated on three benchmarks:
- **ShapeNetPart** — part segmentation (50 parts, 16 categories)
- **ScanObjectNN PB-T50-RS** — real-world object classification (15 classes)
- **S3DIS Area 5** — indoor scene segmentation (13 classes)

Each experiment runs on a single GPU. Three experiments run in parallel on
separate GPUs via SLURM.

---

## Foveated ImageNet Quickstart

The foveated transformer extension is available on `main`.

```bash
git clone https://github.com/AryaPawa/ASP-SNN.git
cd ASP-SNN
```

Important files:
- `models/foveater_asp.py` - FoveaTer-style image ASP model
- `train_imagenet_foveater.py` - ImageNet/ImageNet-100 training script
- `configs/imagenet_foveater.yaml` - default 500-epoch training config
- `datasets/imagenet.py` - ImageFolder dataloader
- `scripts/run_imagenet_foveater.sh` - shell helper

Smoke test the foveated path without ImageNet:

```bash
python smoke_test.py
python train_imagenet_foveater.py \
  --config configs/imagenet_foveater.yaml \
  --set smoke=true epochs=1 batch_size=2 image_size=64 feature_grid=4 \
        embed_dim=48 depth=1 max_fixations=2 max_tokens=8 debug_steps=1 \
        num_workers=0 use_amp=false num_classes=10
```

Train on ImageNet:

```bash
python train_imagenet_foveater.py \
  --config configs/imagenet_foveater.yaml \
  --set data_dir=/path/to/imagenet num_classes=1000 batch_size=128 epochs=500
```

Train on ImageNet-100:

```bash
python train_imagenet_foveater.py \
  --config configs/imagenet_foveater.yaml \
  --set data_dir=/path/to/imagenet100 num_classes=100 batch_size=128 epochs=500
```

Expected dataset layout is standard `torchvision.datasets.ImageFolder`:

```text
data/imagenet/
  train/class_name_or_wnid/*.JPEG
  val/class_name_or_wnid/*.JPEG
```

---

## Environment And Self-Check

You do **not** have to create a new conda environment if your current
environment already has the required packages. If you are already using an
environment for Akarsh's code, try these checks first from the repo root:

```bash
cd ASP-SNN
python --version
python -c "import torch, torchvision, yaml, numpy; print(torch.__version__); print(torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
python smoke_test.py
```

If `python smoke_test.py` prints `ALL 8 TESTS PASSED`, the current environment
is good for code sanity checks and CPU smoke runs.

For real GPU training, also make sure CUDA PyTorch is visible:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

If that prints `False` or `0`, the code can still run CPU smoke tests, but
full training should be run in a CUDA-enabled environment.

If your current environment fails imports or does not have CUDA PyTorch, create
the provided environment:

```bash
cd ASP-SNN
bash setup.sh
conda activate asp-snn
python smoke_test.py
```

### Checks Already Run By Us

From `ASP-SNN/`, on the current checkout, we verified:

```bash
python smoke_test.py
```

Result: `ALL 8 TESTS PASSED`

We also verified the FoveaTer/ImageNet synthetic smoke-training path:

```bash
python train_imagenet_foveater.py --config configs/imagenet_foveater.yaml \
  --set smoke=true epochs=1 batch_size=2 image_size=64 feature_grid=4 \
        embed_dim=48 depth=1 max_fixations=2 max_tokens=8 debug_steps=1 \
        num_workers=0 use_amp=false num_classes=10 verbose_every=1
```

Result: completed one synthetic training/eval pass successfully.

Local limitation: this verification machine had CPU-only PyTorch
(`torch.cuda.is_available() == False`), so GPU execution and real-dataset
training must be checked on the target GPU machine after datasets are present.

---

## Exact Run Commands By Dataset

Run every command in this section from the repository root folder:

```bash
git clone https://github.com/AryaPawa/ASP-SNN.git
cd ASP-SNN
```

If your current environment passed the checks above, keep using it. Otherwise
create and activate the provided environment:

```bash
bash setup.sh
conda activate asp-snn
```

### No-Dataset CPU Smoke Test

From folder: `ASP-SNN/`

```bash
python smoke_test.py
```

Expected result:

```text
ALL 8 TESTS PASSED
```

### ShapeNetPart Segmentation

From folder: `ASP-SNN/`

Download ShapeNetPart:

```bash
python datasets/download.py --shapenet
```

If the Stanford link is slow/down, use the Kaggle dataset
`majdouline20/shapenetpart-dataset`. That Kaggle copy is the raw
`PartAnnotation` layout, so the downloader converts it to the HDF5 format
expected by `train_shapenet.py`:

```bash
python datasets/download.py --shapenet_kaggle
```

If the Kaggle data is already downloaded/extracted:

```bash
python datasets/download.py --shapenet_raw /path/to/PartAnnotation_or_parent_folder
```

Quick 2-epoch smoke training:

```bash
python train_shapenet.py --config configs/shapenet_seg.yaml \
  --set epochs=2 batch_size=4 num_workers=0 eval_interval=1
```

Full training:

```bash
bash scripts/train_shapenet_full.sh
```

Evaluate:

```bash
python eval_shapenet.py --ckpt checkpoints/shapenet_best.pt --config configs/shapenet_seg.yaml --per_cat
```

### ScanObjectNN Classification

From folder: `ASP-SNN/`

Download ScanObjectNN:

```bash
python datasets/download.py --scanobj
```

Quick 2-epoch smoke training:

```bash
python train_scanobj.py --config configs/scanobj_cls.yaml \
  --set epochs=2 batch_size=4 num_workers=0 eval_interval=1
```

Full training:

```bash
bash scripts/train_scanobj_full.sh
```

Evaluate:

```bash
python eval_scanobj.py --ckpt checkpoints/scanobj_best.pt --config configs/scanobj_cls.yaml --n_votes 1
```

Use `--n_votes 10` only as an optional TTA check after the no-TTA number is
healthy. If an older run produced about 30-40% test accuracy after high train
accuracy, discard that checkpoint and retrain; the current code recomputes
geometry after train augmentation and TTA.

### S3DIS Semantic Segmentation

From folder: `ASP-SNN/`

Download S3DIS:

```bash
python datasets/download.py --s3dis
```

Quick 2-epoch smoke training:

```bash
python train_s3dis.py --config configs/s3dis_seg.yaml \
  --set epochs=2 batch_size=4 num_workers=0 eval_interval=1
```

Full training:

```bash
bash scripts/train_s3dis_full.sh
```

The default S3DIS protocol is Area-5 testing: the model trains on Areas
1,2,3,4,6 and evaluates on Area 5. The header now prints both lists. The
default S3DIS batch size is 64 for H100-style 80 GB GPUs; lower it with
`BATCH_SIZE=32 bash scripts/train_s3dis_full.sh` if needed.

Evaluate:

```bash
python eval_s3dis.py --ckpt checkpoints/s3dis_best.pt --config configs/s3dis_seg.yaml --per_class
```

### ImageNet Or ImageNet-100 FoveaTer ASP

From folder: `ASP-SNN/`

First arrange data manually in ImageFolder format:

```text
/path/to/imagenet/
  train/class_name_or_wnid/*.JPEG
  val/class_name_or_wnid/*.JPEG
```

Synthetic smoke training without ImageNet:

```bash
python train_imagenet_foveater.py --config configs/imagenet_foveater.yaml \
  --set smoke=true epochs=1 batch_size=2 image_size=64 feature_grid=4 \
        embed_dim=48 depth=1 max_fixations=2 max_tokens=8 debug_steps=1 \
        num_workers=0 use_amp=false num_classes=10
```

Train on full ImageNet:

```bash
python train_imagenet_foveater.py --config configs/imagenet_foveater.yaml \
  --set data_dir=/path/to/imagenet num_classes=1000 batch_size=128 epochs=500
```

Train on ImageNet-100:

```bash
python train_imagenet_foveater.py --config configs/imagenet_foveater.yaml \
  --set data_dir=/path/to/imagenet100 num_classes=100 batch_size=128 epochs=500
```

### Parallel GPU Smoke Training

From folder: `ASP-SNN/`

Run quick smoke-training jobs in parallel on GPUs 0, 1, 2, and 3:

```bash
bash scripts/smoke_train_parallel.sh
```

Choose GPU IDs explicitly:

```bash
GPU_SHAPENET=0 GPU_SCANOBJ=1 GPU_S3DIS=2 GPU_FOVEATER=3 \
  bash scripts/smoke_train_parallel.sh
```

Manual parallel full training:

```bash
GPU_SHAPENET=0 GPU_SCANOBJ=1 GPU_S3DIS=2 bash scripts/train_pointclouds_parallel.sh &
PID_PC=$!
CUDA_VISIBLE_DEVICES=3 python train_imagenet_foveater.py --config configs/imagenet_foveater.yaml \
  --set data_dir=/path/to/imagenet num_classes=1000 epochs=500 &
PID_IMG=$!
wait "$PID_PC"
wait "$PID_IMG"
```

---

## Architecture

```
Input: point cloud [N points, C channels]
        │
        ▼
┌─────────────────────────────────────────────────┐
│  SLICING: FPS (M=16 anchors) → KNN (K per anchor) │
│  + 8-dim geometry descriptor per slice            │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  ENCODER: EdgeConv (static kNN, k=20)            │
│  Conv2d(2C→128→128) + Conv1d(128→256→512)       │
│  Global max-pool → one 512-dim token per slice   │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  POSITIONAL ENCODING: Linear(3→512) on centroid  │
│  + TRANSFORMER: 1 layer, 4 heads, FFN=1024       │
│  Output: [B, 16, 512] contextualised tokens      │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  ASP LOOP (T=6 timesteps):                       │
│  ├─ SSP: attention-based slice scoring           │
│  │   (belief × geometry → score per slice)       │
│  ├─ Gumbel-softmax (train) / argmax (eval)       │
│  ├─ Fuse: selected feature + belief projection   │
│  ├─ 3-layer LIF (soft reset, ATan surrogate):    │
│  │   Linear→BN→ReLU→LIF dynamics + residual      │
│  ├─ Emit logits (classification) or              │
│  │   accumulate belief (segmentation)             │
│  └─ Early exit if confidence > threshold (eval)   │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  OUTPUT:                                         │
│  Classification: logit averaging across T steps  │
│  Segmentation: per-point features via slice      │
│    lookup + PerPointBranch + global belief +     │
│    SegHead MLP → [B, N, num_classes]             │
└─────────────────────────────────────────────────┘
```

Key architectural parameters:
- `feat_dim = hidden_dim = 512`
- `num_slices (M) = 16`, `points_per_slice (K) = 128` (ShapeNet/ScanObj) or `256` (S3DIS)
- `num_lif_layers = 3`, `lif_leak = 0.9`, `lif_threshold = 1.0`
- `T = 6` ASP timesteps, `exit_threshold = 0.40`
- Surrogate gradient: ATan (α=2.0) per Fang et al. ICCV 2021 (SPM-compatible)

---

## Repository structure

```
ASP-SNN/
├── README.md                       # This file
├── environment.yml                 # Conda environment specification
├── requirements.txt                # Pip dependencies
├── setup.sh                        # One-command environment setup
├── smoke_test.py                   # 8-test verification suite (CPU, no data)
│
├── configs/                        # Per-dataset YAML configurations
│   ├── shapenet_seg.yaml           #   500 epochs, bs=32, pts_per_slice=128
│   ├── scanobj_cls.yaml            #   500 epochs, bs=32, SO(3) aug, SWA, deep MLP head
│   ├── imagenet_foveater.yaml      #   FoveaTer ASP on ImageNet/ImageNet-100
│   └── s3dis_seg.yaml              #   500 epochs, bs=64, RGB+height, class weights
│
├── scripts/                        # SLURM job scripts for GPU cluster
│   ├── run_shapenet.sh             #   18h wall, 1 GPU, 8 CPUs, 32G
│   ├── run_scanobj.sh              #   8h wall, 1 GPU, 8 CPUs, 32G
│   ├── run_s3dis.sh                #   72h wall, 1 GPU, 8 CPUs, 64G
│   ├── run_imagenet_foveater.sh    #   FoveaTer ASP ImageNet helper
│   ├── train_shapenet_full.sh      #   Shell-first 500-epoch ShapeNetPart run
│   ├── train_scanobj_full.sh       #   Shell-first 500-epoch ScanObjectNN run
│   ├── train_s3dis_full.sh         #   Shell-first 500-epoch S3DIS run
│   ├── train_pointclouds_parallel.sh # Parallel full point-cloud trainings
│   ├── smoke_train.sh              #   Quick serial 2-epoch test for all 3 point-cloud datasets
│   └── smoke_train_parallel.sh     #   Quick parallel smoke jobs across GPUs
│
├── config.py                       # YAML loader with auto type casting + CLI overrides
│
├── datasets/                       # Data loading and preprocessing
│   ├── __init__.py
│   ├── download.py                 # Downloads all 3 datasets (auto where possible)
│   ├── shapenetpart.py             # ShapeNetPart HDF5 loader (14,007 train / 2,874 test)
│   ├── scanobjectnn.py             # ScanObjectNN PB-T50-RS loader (11,416 train / 2,882 test)
│   ├── s3dis.py                    # S3DIS room-block loader (6 areas, 271 rooms)
│   ├── imagenet.py                 # ImageFolder loader for ImageNet/ImageNet-100
│   ├── slicing.py                  # FPS + KNN slicing + 8-dim geometry descriptors
│   └── transforms.py               # Augmentation: rotation, scale, jitter, color dropout
│
├── models/                         # Network architecture
│   ├── __init__.py
│   ├── encoder.py                  # EdgeConv feature extractor (variable input channels)
│   ├── ssp.py                      # Slice Selection Policy (attention-based scoring)
│   ├── lif.py                      # Multi-layer LIF with ATan surrogate (AMP-safe)
│   ├── asp_classifier.py           # ASP model for classification (ScanObjectNN)
│   ├── asp_segmentor.py            # ASP model for segmentation (ShapeNet + S3DIS)
│   └── foveater_asp.py             # FoveaTer-style image ASP model
│
├── train_shapenet.py               # Train ShapeNetPart (differential LR, category masking)
├── train_scanobj.py                # Train ScanObjectNN (SWA, aux loss, label smoothing)
├── train_s3dis.py                  # Train S3DIS (class-weighted loss, RGB+height)
├── train_imagenet_foveater.py      # Train FoveaTer ASP on ImageNet/ImageNet-100
├── eval_shapenet.py                # Eval ShapeNetPart (per-category IoU)
├── eval_scanobj.py                 # Eval ScanObjectNN (N-vote TTA, per-class accuracy)
├── eval_s3dis.py                   # Eval S3DIS (per-class IoU/Acc, mIoU, OA)
│
├── checkpoints/                    # Saved models (auto-created)
├── logs/                           # CSV training logs (auto-created)
└── data/                           # Datasets (auto-created by download.py)
```

---

## Setup (one-time, ~5 minutes)

### Prerequisites
- conda or miniconda installed
- CUDA 12.1+ compatible GPU drivers
- Internet access for pip install and dataset download

### Commands

```bash
# 1. Clone the repository
git clone https://github.com/AryaPawa/ASP-SNN.git
cd ASP-SNN

# 2. Create conda environment and install all dependencies
bash setup.sh

# 3. Activate environment
conda activate asp-snn

# 4. Verify installation (CPU only, no data needed, ~30 seconds)
python smoke_test.py
# Expected output: ALL 8 TESTS PASSED
```

The smoke test verifies:
1. ASPClassifier builds and trains (forward + backward)
2. ASPSegmentor works in ShapeNetPart mode (50 classes, category conditioning)
3. ASPSegmentor works in S3DIS mode (13 classes, 7-channel input)
4. FPS + KNN slicing pipeline produces correct shapes
5. Early exit does not crash
6. Batch size 1 evaluation works (BN edge case)
7. FoveaTer ASP image model builds and trains
8. Checkpoint save/load preserves predictions

---

## FoveaTer ASP on ImageNet

This repo now includes an image-domain ASP model based on the FoveaTer
architecture. It uses a CNN feature map, fixation-dependent square foveated
pooling, transformer class-token attention for guided next fixation, inhibition
of return, average pooling across fixation states, and confidence-based early
exit.

### Quick synthetic smoke run

This does not need ImageNet and is useful after setup:

```bash
python train_imagenet_foveater.py \
  --config configs/imagenet_foveater.yaml \
  --set smoke=true epochs=1 batch_size=2 image_size=64 feature_grid=4 \
        embed_dim=48 depth=1 max_fixations=2 max_tokens=8 debug_steps=1 \
        num_workers=0 use_amp=false num_classes=10
```

### ImageNet folder layout

Use standard `torchvision.datasets.ImageFolder` layout:

```text
data/imagenet/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

ImageNet-100 uses the same layout with 100 class folders.

### Train on ImageNet

```bash
python train_imagenet_foveater.py \
  --config configs/imagenet_foveater.yaml \
  --set data_dir=/path/to/imagenet num_classes=1000 batch_size=128 epochs=500
```

For ImageNet-100:

```bash
python train_imagenet_foveater.py \
  --config configs/imagenet_foveater.yaml \
  --set data_dir=/path/to/imagenet100 num_classes=100 batch_size=128 epochs=500
```

You can also use the helper:

```bash
DATA_DIR=/path/to/imagenet NUM_CLASSES=1000 EPOCHS=500 bash scripts/run_imagenet_foveater.sh
```

Checkpoints are written to `checkpoints/imagenet_foveater_last.pt` and
`checkpoints/imagenet_foveater_best.pt`. CSV logs are written under `logs/`.

---

## Dataset download

```bash
# Download all three datasets
python datasets/download.py --all

# Or individually:
python datasets/download.py --shapenet    # ~346 MB from Stanford
python datasets/download.py --scanobj     # ~50 MB from HuggingFace
python datasets/download.py --s3dis       # ~15.5 GB via gdown/OpenPoints
```

For the Kaggle ShapeNetPart dataset:

```bash
python datasets/download.py --shapenet_kaggle
```

The specific Kaggle dataset `majdouline20/shapenetpart-dataset` is a raw
`PartAnnotation` tree, not the same HDF5 file layout as Stanford. The command
above downloads and converts it. If already downloaded/extracted:

```bash
python datasets/download.py --shapenet_raw /path/to/PartAnnotation_or_parent_folder
```

| Dataset | Source | Auth required? | Auto-download? |
|---|---|---|---|
| ShapeNetPart | Stanford direct, then Kaggle `majdouline20/shapenetpart-dataset` | Kaggle may need API credentials | Yes |
| ScanObjectNN | `huggingface.co/datasets/cminst/ScanObjectNN` | No | Yes |
| S3DIS | Google Drive (OpenPoints preprocessed) | No | Yes (via gdown) |

If S3DIS gdown fails (rate limiting), follow the printed manual instructions or:
```bash
# Option A: Retry with gdown
pip install gdown
python datasets/download.py --s3dis

# Option B: Preprocess raw S3DIS from Stanford
python datasets/download.py --s3dis_preprocess /path/to/Stanford3dDataset_v1.2_Aligned_Version
```

Expected data layout after download:
```
data/
├── shapenet_part_seg_hdf5_data/    # train0-5.h5, test0-1.h5
├── ScanObjectNN/main_split/        # training_objectdataset_augmentedrot_scale75.h5
│                                   # test_objectdataset_augmentedrot_scale75.h5
└── s3dis/Area_1/ ... Area_6/       # per-room .npy files [N, 7]: x,y,z,r,g,b,label
# or: s3dis/raw/Area_1_*.npy ... Area_6_*.npy
```

---

## Training

### Option A: SLURM cluster (recommended)

Edit the `#SBATCH` headers in `scripts/run_*.sh` to match your cluster's
partition and account names, then submit:

```bash
sbatch scripts/run_shapenet.sh     # ~12-13h on H100 for 500 epochs
sbatch scripts/run_scanobj.sh      # ~4-5h on H100 for 500 epochs
sbatch scripts/run_s3dis.sh        # ~18-28h on H100 for 500 epochs at batch 64
```

All three run in parallel on separate GPUs.

### Option B: Interactive (3 terminals)

```bash
GPU_SHAPENET=0 GPU_SCANOBJ=1 GPU_S3DIS=2 bash scripts/train_pointclouds_parallel.sh
```

### Quick smoke training (verify full pipeline with real data, ~10 min)

```bash
bash scripts/smoke_train.sh
# Runs 2 epochs per dataset with batch_size=4
```

### Resume from crash

All scripts automatically save `*_last.pt` every epoch. To resume:

```bash
python train_scanobj.py --config configs/scanobj_cls.yaml --resume checkpoints/scanobj_last.pt
python train_shapenet.py --config configs/shapenet_seg.yaml --resume checkpoints/shapenet_last.pt
python train_s3dis.py --config configs/s3dis_seg.yaml --resume checkpoints/s3dis_last.pt
```

Resumes model weights, optimizer state, LR scheduler, AMP scaler, and epoch counter.

### Override config values from CLI

```bash
python train_scanobj.py --config configs/scanobj_cls.yaml --set epochs=500 lr=1e-3 batch_size=32
```

### Monitor training

```bash
squeue -u $USER                      # SLURM job status
tail -f logs/scanobj_*.log           # Live SLURM output
cat logs/scanobj_*.csv               # CSV: epoch, loss, train_acc, val_acc, test_acc, lr, time
```

---

## Evaluation

```bash
# ScanObjectNN - first check no-TTA accuracy
python eval_scanobj.py --ckpt checkpoints/scanobj_best.pt --n_votes 1

# ShapeNetPart — with per-category IoU breakdown
python eval_shapenet.py --ckpt checkpoints/shapenet_best.pt --per_cat

# S3DIS — with per-class IoU and accuracy
python eval_s3dis.py --ckpt checkpoints/s3dis_best.pt --per_class
```

---

## Expected results

| Task | Dataset | Metric | Expected | PointNet++ | DGCNN | SPM (SNN) |
|---|---|---|---|---|---|---|
| Classification | ScanObjectNN PB-T50-RS | OA | 82-87% | 77.9% | 78.1% | 84.2% |
| Part seg | ShapeNetPart | Inst mIoU | 81-84% | 85.1% | 85.2% | 84.8% |
| Scene seg | S3DIS Area 5 | mIoU | 50-58% | 53.5% | 56.1% | — |

Estimated training time on NVIDIA H100 80GB:

| Dataset | Epochs | Batch size | Time per epoch | Total |
|---|---|---|---|---|
| ScanObjectNN | 500 | 32 | ~30s | ~4-5h |
| ShapeNetPart | 500 | 32 | ~1.5min | ~12-13h |
| S3DIS | 500 | 64 | ~2-3min | ~18-28h |

---

## Output structure (after training)

```
checkpoints/
├── shapenet_best.pt      # Best ShapeNetPart model (by Instance mIoU)
├── shapenet_last.pt      # Last epoch (for --resume)
├── scanobj_best.pt       # Best ScanObjectNN model (by validation accuracy)
├── scanobj_last.pt       # Last epoch
├── scanobj_swa.pt        # SWA averaged model (ScanObjectNN only)
├── s3dis_best.pt         # Best S3DIS model (by mIoU)
└── s3dis_last.pt         # Last epoch

logs/
├── shapenet_YYYYMMDD_HHMMSS.csv   # epoch, train_loss, inst_miou, cls_miou, lr, time
├── scanobj_YYYYMMDD_HHMMSS.csv    # epoch, train_loss, train_acc, val_acc, lr, time
└── s3dis_YYYYMMDD_HHMMSS.csv      # epoch, train_loss, miou, macc, oa, lr, time
```

All checkpoints contain: `{epoch, model, optimizer, scheduler, scaler, best_metric}` — fully resumable.

---

## Key design decisions

| Decision | Rationale |
|---|---|
| Static kNN (xyz only) in encoder | Dynamic kNN on 64-point slices caused ±5% val oscillation |
| ATan surrogate gradient (α=2.0) | SPM-compatible, SOTA for SNNs (Fang et al. ICCV 2021) |
| AMP-safe surrogate with `custom_fwd` | Prevents fp16 overflow in pow(2) on H100 with mixed precision |
| Soft reset LIF dynamics | `u_t = λ·u + inp − θ·s` preserves membrane information vs hard reset |
| Deterministic FPS at test time | `seed=idx` prevents non-reproducible eval metrics |
| 3-layer MLP classifier for ScanObjectNN | Single Linear bottleneck on real-world data (+1-2%) |
| Class-weighted CE for S3DIS | Inverse-frequency weighting for severely imbalanced classes |
| RGB + normalised height feature for S3DIS | Height alone gives +3-5% mIoU (floor=0, ceiling=1) |
| Color dropout (p=0.2) for S3DIS | Forces geometry learning; prevents RGB overfitting |
| Aux loss normalised by weight sum | Prevents ~2.58× effective LR inflation from multi-timestep loss |
| SWA only when averaging window ≥ 10 epochs | Avoids meaningless weight averaging on short runs |
| Differential LR for ShapeNet encoder (0.1×) | Stabilises pre-existing encoder features during seg head training |

---

## Research context

**Reference backbone:** Spiking Point Mamba (SPM), arxiv 2504.14371

**What ASP-SNN contributes beyond SPM:**
- First **active perception** framework for SNN-based point clouds (sequential slice selection via SSP)
- First SNN method evaluated across **classification + part segmentation + scene segmentation** in a unified architecture
- **Interpretable** slice selection (SSP attention weights show which regions the model examines)
- **Adaptive compute** via early exit (simple shapes use fewer slices)
- Honest **energy efficiency analysis** with per-layer spike rate logging (`SpikeRateLogger`)

**Competitive landscape (SNN domain):**
- SPM: 84.2% ScanObjectNN, 84.8% ShapeNetPart Inst mIoU
- 3DSMT (ICLR 2026): 90.4% ScanObjectNN, 70.2% S3DIS mIoU
- S3DNet: 84.5% ScanObjectNN

Our positioning: competitive accuracy with SPM while offering fundamentally different capabilities (active perception, early exit, interpretability) that no existing SNN method provides.

---

## Spike rate logging (for energy efficiency metric)

After training, measure per-layer spike rates for the paper's energy analysis:

```python
from models.lif import SpikeRateLogger

model.eval()
logger = SpikeRateLogger()
model.lif_head.spike_monitor = logger

# Run evaluation
evaluate(model, test_loader, device)

print(f"Mean spike rate: {logger.mean_rate():.4f}")
print(f"Per-layer rates: {logger.per_layer_rates()}")

model.lif_head.spike_monitor = None
```

---

## Citation

```bibtex
@article{asp_snn_2026,
    title={Active Spiking Perception for 3D Point Cloud Understanding},
    author={},
    year={2026}
}
```
