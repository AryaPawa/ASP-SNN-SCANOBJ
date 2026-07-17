"""
train_dgcnn_teacher.py — Train a DGCNN teacher on ScanObjectNN, then save its
logits over the training set so ASP-SNN can distill from them without ever
loading the teacher during ASP-SNN training.

Two phases:
  1. Train DGCNN classifier on ScanObjectNN PB-T50-RS train split
  2. Run inference over the training set (no augmentation) and save the
     logits as a tensor of shape [N_train, num_classes] to disk.

Downstream: train_scanobj.py will load teacher_logits and apply KD loss.

Usage:
    python train_dgcnn_teacher.py --epochs 200 --batch_size 32

Output:
    checkpoints/dgcnn_teacher.pt         (best model)
    data/ScanObjectNN/teacher_logits.pt  ({logits: [N, C], indices: [N]})
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from config import load_config, set_seed
from datasets.scanobjectnn import ScanObjectNNDataset
from models.dgcnn_teacher import DGCNNTeacher


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for slices, geo, labels in loader:
        # DGCNN uses raw points, not slices — reconstruct point cloud
        # from the slice tensor: [B, M, K, 6] -> [B, M*K, 3]
        B, M, K, C = slices.shape
        points = slices[..., :3].reshape(B, M * K, 3).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(points)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * B
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples += B

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for slices, geo, labels in loader:
        B, M, K, C = slices.shape
        points = slices[..., :3].reshape(B, M * K, 3).to(device)
        labels = labels.to(device)
        logits = model(points)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += B
    return correct / max(total, 1)


@torch.no_grad()
def dump_train_logits(model, dataset, device, out_path: str,
                      batch_size: int = 32, num_workers: int = 2):
    """
    Run the trained teacher over the training set with NO augmentation,
    and save the logits + sample indices to disk for KD.
    """
    model.eval()
    # Force the dataset into eval-style behavior for consistent logits:
    # We build a NEW dataset instance with force_no_aug=True so the same
    # sample always produces the same logits (deterministic teacher).
    print(f"[Teacher] Dumping training-set logits (no augmentation) ...")

    all_logits = []
    all_indices = []

    # Iterate the dataset in order (no shuffling) so indices are stable
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    idx_start = 0
    for slices, geo, labels in loader:
        B, M, K, C = slices.shape
        points = slices[..., :3].reshape(B, M * K, 3).to(device)
        logits = model(points).cpu()                              # [B, num_classes]
        all_logits.append(logits)
        all_indices.append(torch.arange(idx_start, idx_start + B))
        idx_start += B

    logits_tensor = torch.cat(all_logits, dim=0)                  # [N, C]
    indices_tensor = torch.cat(all_indices, dim=0)                # [N]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        'logits': logits_tensor,
        'indices': indices_tensor,
        'num_samples': int(logits_tensor.size(0)),
        'num_classes': int(logits_tensor.size(1)),
    }, out_path)
    print(f"[Teacher] Saved logits: {tuple(logits_tensor.shape)} -> {out_path}")


def main():
    p = argparse.ArgumentParser(description="Train DGCNN teacher for KD")
    p.add_argument('--config', type=str, default='configs/scanobj_cls.yaml')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--ckpt_out', type=str, default='checkpoints/dgcnn_teacher.pt')
    p.add_argument('--logits_out', type=str,
                   default='data/ScanObjectNN/teacher_logits.pt')
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = cfg.device

    print(f"\n{'='*60}")
    print(f"  DGCNN Teacher Training on ScanObjectNN PB-T50-RS")
    print(f"  Epochs: {args.epochs}  LR: {args.lr}  Batch: {args.batch_size}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── Datasets ──────────────────────────────────────────────────────
    train_ds = ScanObjectNNDataset(cfg.data_dir, 'train', cfg)
    train_ds_noaug = ScanObjectNNDataset(cfg.data_dir, 'train', cfg,
                                         force_no_aug=True)
    test_ds = ScanObjectNNDataset(cfg.data_dir, 'test', cfg)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    # ── Model ──────────────────────────────────────────────────────────
    model = DGCNNTeacher(
        num_classes=cfg.num_classes,
        k=cfg.k_edge,
        emb_dims=1024,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DGCNN parameters: {n_params:,}")

    # ── Optimizer / schedule ──────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
    )
    scaler = GradScaler(device=device.type, enabled=cfg.use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Training loop ─────────────────────────────────────────────────
    best_test = 0.0
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg.use_amp,
        )
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            test_acc = evaluate(model, test_loader, device)
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1:3d}/{args.epochs}] "
                  f"lr={lr_now:.2e} | loss={train_loss:.4f} "
                  f"train={train_acc*100:.1f}% test={test_acc*100:.2f}% "
                  f"| {elapsed:.0f}s", flush=True)

            if test_acc > best_test:
                best_test = test_acc
                torch.save({
                    'model': model.state_dict(),
                    'test_acc': best_test,
                    'epoch': epoch + 1,
                }, args.ckpt_out)
                print(f"  >> New best test: {best_test*100:.2f}%", flush=True)
        else:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1:3d}/{args.epochs}] "
                  f"lr={lr_now:.2e} | loss={train_loss:.4f} "
                  f"train={train_acc*100:.1f}% | {elapsed:.0f}s", flush=True)

    print(f"\nBest test accuracy: {best_test*100:.2f}%")

    # ── Load best model and dump logits ───────────────────────────────
    print(f"\nLoading best teacher: {args.ckpt_out}")
    ckpt = torch.load(args.ckpt_out, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])

    # Dump training-set logits with NO augmentation for consistency
    dump_train_logits(model, train_ds_noaug, device, args.logits_out,
                      batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"\nTeacher training + logit dump complete.")


if __name__ == '__main__':
    main()