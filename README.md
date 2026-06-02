# ASP-SNN: Active Spiking Perception for 3D Point Cloud Understanding

This repository contains the code for training and evaluating ASP-SNN on three
benchmark datasets: **ShapeNetPart** (part segmentation), **ScanObjectNN**
(classification), and **S3DIS** (scene segmentation).

All experiments are designed to run on a single GPU each (tested on NVIDIA H100
80 GB). Three experiments run in parallel on separate GPUs.

---

## Quick start

```bash
# 1. Create environment and install dependencies
bash setup.sh

# 2. Activate the environment
conda activate asp-snn

# 3. Download datasets
python datasets/download.py --all

# 4. Run all three experiments (3 GPUs)
#    Option A: interactive terminals
CUDA_VISIBLE_DEVICES=0 python train_shapenet.py &
CUDA_VISIBLE_DEVICES=1 python train_scanobj.py &
CUDA_VISIBLE_DEVICES=2 python train_s3dis.py &

#    Option B: SLURM cluster (recommended)
sbatch scripts/run_shapenet.sh
sbatch scripts/run_scanobj.sh
sbatch scripts/run_s3dis.sh
```

---

## Repository structure

```
ASP-SNN/
├── README.md                   # This file
├── environment.yml             # Conda environment specification
├── requirements.txt            # Pip dependencies
├── setup.sh                    # One-command setup
│
├── configs/                    # Per-dataset YAML configurations
│   ├── shapenet_seg.yaml
│   ├── scanobj_cls.yaml
│   └── s3dis_seg.yaml
│
├── scripts/                    # SLURM job scripts
│   ├── run_shapenet.sh
│   ├── run_scanobj.sh
│   └── run_s3dis.sh
│
├── datasets/                   # Data loading and preprocessing
│   ├── __init__.py
│   ├── download.py             # Downloads all three datasets
│   ├── shapenetpart.py         # ShapeNetPart HDF5 loader
│   ├── scanobjectnn.py         # ScanObjectNN HDF5 loader
│   ├── s3dis.py                # S3DIS room-block loader
│   ├── slicing.py              # FPS + KNN slicing + geometry descriptors
│   └── transforms.py           # Data augmentation functions
│
├── models/                     # Network architecture
│   ├── __init__.py
│   ├── encoder.py              # EdgeConv feature extractor
│   ├── ssp.py                  # Slice Selection Policy
│   ├── lif.py                  # Multi-layer LIF temporal head
│   ├── asp_classifier.py       # ASP model for classification tasks
│   └── asp_segmentor.py        # ASP model for segmentation tasks
│
├── train_shapenet.py           # Train ShapeNetPart part segmentation
├── train_scanobj.py            # Train ScanObjectNN classification
├── train_s3dis.py              # Train S3DIS scene segmentation
├── eval_shapenet.py            # Evaluate ShapeNetPart
├── eval_scanobj.py             # Evaluate ScanObjectNN
├── eval_s3dis.py               # Evaluate S3DIS
│
├── checkpoints/                # Saved model weights (auto-created)
├── logs/                       # CSV training logs (auto-created)
└── data/                       # Datasets (auto-created by download.py)
```

---

## Datasets

| Dataset | Task | Classes | Train / Test | Points | Download |
|---|---|---|---|---|---|
| ShapeNetPart | Part segmentation | 50 parts, 16 categories | 14,007 / 2,874 | 2,048 | Auto (Stanford) |
| ScanObjectNN PB-T50-RS | Classification | 15 | 11,416 / 2,882 | 2,048 | Manual (see below) |
| S3DIS Area 5 | Scene segmentation | 13 | Areas 1,2,3,4,6 / Area 5 | 4,096/block | Auto (HuggingFace) |

**ScanObjectNN** requires accepting a license agreement before download.
Visit `https://hkust-vgd.github.io/scanobjectnn/` and fill out the form to
obtain the download link. Then place the files as:

```
data/ScanObjectNN/main_split/
    training_objectdataset_augmentedrot_scale75.h5
    test_objectdataset_augmentedrot_scale75.h5
```

Alternatively, if you have the download link:
```bash
python datasets/download.py --scanobj_url "YOUR_DOWNLOAD_URL"
```

---

## Training

Each training script is fully standalone. Configuration is loaded from the
corresponding YAML file and can be overridden via command-line flags.

```bash
# ShapeNetPart (~8 h on H100)
python train_shapenet.py [--config configs/shapenet_seg.yaml] [--resume checkpoints/shapenet_best.pt]

# ScanObjectNN (~4 h on H100)
python train_scanobj.py [--config configs/scanobj_cls.yaml] [--resume checkpoints/scanobj_best.pt]

# S3DIS (~12 h on H100)
python train_s3dis.py [--config configs/s3dis_seg.yaml] [--resume checkpoints/s3dis_best.pt]
```

All scripts save checkpoints to `checkpoints/` and CSV logs to `logs/`.
Training automatically resumes from the last checkpoint if `--resume` is passed.

---

## Evaluation

```bash
python eval_shapenet.py --ckpt checkpoints/shapenet_best.pt --per_cat
python eval_scanobj.py  --ckpt checkpoints/scanobj_best.pt
python eval_s3dis.py    --ckpt checkpoints/s3dis_best.pt --per_class
```

---

## SLURM cluster usage

SLURM scripts are provided in `scripts/`. Edit the `#SBATCH` headers to match
your cluster partition and account. Then:

```bash
sbatch scripts/run_shapenet.sh
sbatch scripts/run_scanobj.sh
sbatch scripts/run_s3dis.sh
```

Monitor progress:
```bash
squeue -u $USER
tail -f logs/shapenet_*.log
```

---

## Expected results

| Task | Dataset | Metric | Target | PointNet++ | DGCNN | SPM (SNN) |
|---|---|---|---|---|---|---|
| Part seg | ShapeNetPart | Inst mIoU | 83-85% | 85.1% | 85.2% | 84.8% |
| Classification | ScanObjectNN | OA | 85-88% | 77.9% | 78.1% | 84.2% |
| Scene seg | S3DIS Area 5 | mIoU | 55-62% | 53.5% | 56.1% | — |

---

## Citation

```bibtex
@article{asp_snn_2026,
  title={Active Spiking Perception for 3D Point Cloud Understanding},
  author={...},
  year={2026}
}
```
