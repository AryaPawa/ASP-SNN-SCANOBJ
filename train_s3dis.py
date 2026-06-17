"""
train_s3dis.py — Train ASP-SNN on S3DIS Area 5 scene segmentation.

Usage:
    python train_s3dis.py [--config configs/s3dis_seg.yaml] [--resume ckpt.pt]
"""

import math
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from config import load_config, set_seed, base_argparser, parse_overrides
from datasets.s3dis import S3DISDataset, CLASS_NAMES, NUM_CLASSES, compute_class_weights
from models.asp_segmentor import ASPSegmentor


def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int):
    """Compute per-class IoU, mIoU, OA, and mAcc."""
    ious = []
    accs = []
    for c in range(num_classes):
        pred_c = (pred == c)
        true_c = (target == c)
        inter = np.logical_and(pred_c, true_c).sum()
        union = np.logical_or(pred_c, true_c).sum()
        if union > 0:
            ious.append(inter / union)
        else:
            ious.append(float('nan'))
        true_count = true_c.sum()
        if true_count > 0:
            accs.append(inter / true_count)
        else:
            accs.append(float('nan'))

    iou_arr = np.array(ious)
    acc_arr = np.array(accs)
    miou = float(np.nanmean(iou_arr))
    macc = float(np.nanmean(acc_arr))
    oa = float((pred == target).sum() / max(len(target), 1))
    return miou, macc, oa, {CLASS_NAMES[i]: ious[i] for i in range(num_classes)}


def main():
    parser = base_argparser("ASP-SNN S3DIS Training")
    args = parser.parse_args()
    overrides = parse_overrides(args)

    config_path = args.config or "configs/s3dis_seg.yaml"
    cfg = load_config(config_path, overrides)
    set_seed(cfg.seed)
    device = cfg.device

    test_area = getattr(cfg, 'test_area', 5)
    train_areas_str = ", ".join(str(a) for a in [1, 2, 3, 4, 5, 6] if a != test_area)

    print(f"\n{'='*60}")
    print(f"  ASP-SNN S3DIS Scene Segmentation")
    print(f"  Protocol: train on Areas {{{train_areas_str}}}, test on Area {test_area}")
    print(f"  Epochs: {cfg.epochs}  LR: {cfg.lr}  Batch: {cfg.batch_size}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── Datasets ──────────────────────────────────────────────────────
    train_ds = S3DISDataset(cfg.data_dir, 'train', cfg)
    test_ds = S3DISDataset(cfg.data_dir, 'test', cfg)

    pw = cfg.num_workers > 0
    # drop_last safety: only drop if we have enough samples to spare
    drop_last = len(train_ds) >= cfg.batch_size * 2
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        drop_last=drop_last, persistent_workers=pw,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=pw,
    )

    # ── Class weights ─────────────────────────────────────────────────
    class_weights = None
    if getattr(cfg, 'use_class_weights', True):
        print("Computing class weights from training areas ...")
        weights_np = compute_class_weights(cfg.data_dir,
                                           getattr(cfg, 'test_area', 5))
        class_weights = torch.from_numpy(weights_np).to(device)
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name:<12} weight={weights_np[i]:.3f}")

    # ── Model ─────────────────────────────────────────────────────────
    # Determine input channels from config
    in_ch = 3
    if getattr(cfg, 'use_rgb', True):
        in_ch += 3
    if getattr(cfg, 'use_height', True):
        in_ch += 1
    cfg.in_channels = in_ch
    cfg.num_classes = NUM_CLASSES
    cfg.use_category = False
    cfg.num_categories = 0

    model = ASPSegmentor(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Input channels: {in_ch} (xyz"
          f"{'+rgb' if getattr(cfg, 'use_rgb', True) else ''}"
          f"{'+height' if getattr(cfg, 'use_height', True) else ''})")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    def lr_lambda(epoch):
        warmup = getattr(cfg, 'warmup_epochs', 5)
        if epoch < warmup:
            return 0.1 + 0.9 * (epoch / warmup)
        progress = (epoch - warmup) / max(1, cfg.epochs - warmup)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=cfg.use_amp)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=-1,  # safety: if any invalid labels exist
    )

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 0
    best_miou = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt.get('epoch', 0)
        best_miou = ckpt.get('best_metric', 0.0)
        print(f"Resumed from epoch {start_epoch}, best mIoU: {best_miou*100:.2f}%")

    # ── Logging ───────────────────────────────────────────────────────
    run_name = f"s3dis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = os.path.join(cfg.log_dir, f"{run_name}.csv")
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,miou,macc,oa,lr,time\n")

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        tau = max(cfg.tau_end, cfg.tau_start * (cfg.tau_decay ** epoch))
        model.gumbel_tau.fill_(tau)

        # ── Train ─────────────────────────────────────────────────────
        model.train()
        total_loss = n_batches = 0
        n_total_batches = len(train_loader)
        log_every = max(1, n_total_batches // 20)  # ~20 progress prints per epoch

        for batch_idx, (slices, geo, pts_feat, sid_arr, sem_labels, cat_ids) in enumerate(train_loader):
            slices = slices.to(device, non_blocking=True)
            geo = geo.to(device, non_blocking=True)
            pts_feat = pts_feat.to(device, non_blocking=True)
            sid_arr = sid_arr.to(device, non_blocking=True)
            sem_labels = sem_labels.to(device, non_blocking=True)
            cat_ids = cat_ids.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=cfg.use_amp):
                logits, _ = model(
                    slices, geo, sid_arr, cat_ids, pts_feat, training=True
                )
                # logits: [B, N, 13]
                B, N, C = logits.shape
                loss = criterion(
                    logits.reshape(B * N, C),
                    sem_labels.reshape(B * N),
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

            # Per-batch progress with ETA — fixes "no console output for 8 minutes"
            if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_total_batches:
                elapsed = time.time() - t0
                per_batch = elapsed / (batch_idx + 1)
                remaining = per_batch * (n_total_batches - batch_idx - 1)
                gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                print(
                    f"  ep{epoch+1} [{batch_idx+1:4d}/{n_total_batches}] "
                    f"loss={loss.item():.4f} eta={remaining:.0f}s "
                    f"gpu_mem={gpu_mem:.1f}GB",
                    flush=True,
                )

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)
        lr_now = optimizer.param_groups[0]['lr']

        # ── Eval ──────────────────────────────────────────────────────
        eval_interval = getattr(cfg, 'eval_interval', 5)
        if (epoch + 1) % eval_interval == 0 or epoch == cfg.epochs - 1:
            model.eval()
            all_preds, all_true = [], []

            with torch.no_grad():
                for slices, geo, pts_feat, sid_arr, sem_labels, cat_ids in test_loader:
                    slices = slices.to(device)
                    geo = geo.to(device)
                    pts_feat = pts_feat.to(device)
                    sid_arr = sid_arr.to(device)
                    cat_ids = cat_ids.to(device)

                    logits, _ = model(
                        slices, geo, sid_arr, cat_ids, pts_feat, training=False
                    )
                    preds = logits.argmax(dim=-1)  # [B, N]
                    all_preds.append(preds.cpu().numpy().reshape(-1))
                    all_true.append(sem_labels.numpy().reshape(-1))

            all_preds = np.concatenate(all_preds)
            all_true = np.concatenate(all_true)
            miou, macc, oa, per_class = compute_iou(
                all_preds, all_true, NUM_CLASSES
            )

            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch+1:3d}/{cfg.epochs}] "
                f"tau={tau:.3f} lr={lr_now:.2e} | "
                f"loss={train_loss:.4f} | "
                f"mIoU={miou*100:.2f}% mAcc={macc*100:.2f}% "
                f"OA={oa*100:.2f}% | {elapsed:.0f}s"
            )

            if (epoch + 1) % 25 == 0 or epoch == cfg.epochs - 1:
                for name, iou in sorted(per_class.items(),
                                        key=lambda x: x[1] if not np.isnan(x[1]) else 0):
                    v = iou * 100 if not np.isnan(iou) else 0.0
                    print(f"    {name:<12} {v:5.1f}%")

            if miou > best_miou:
                best_miou = miou
                torch.save({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_metric': best_miou,
                    'miou': miou,
                    'macc': macc,
                    'oa': oa,
                }, os.path.join(cfg.ckpt_dir, 's3dis_best.pt'))
                print(f"    >> New best mIoU: {miou*100:.2f}%")

            with open(log_path, 'a') as f:
                f.write(f"{epoch+1},{train_loss:.4f},"
                        f"{miou*100:.2f},{macc*100:.2f},{oa*100:.2f},"
                        f"{lr_now:.2e},{elapsed:.0f}\n")
        else:
            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch+1:3d}/{cfg.epochs}] "
                f"tau={tau:.3f} lr={lr_now:.2e} | "
                f"loss={train_loss:.4f} | {elapsed:.0f}s"
            )

        # Save last for resume
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_metric': best_miou,
        }, os.path.join(cfg.ckpt_dir, 's3dis_last.pt'))

    print(f"\nDone. Best mIoU: {best_miou*100:.2f}%")
    print(f"Checkpoint: {cfg.ckpt_dir}/s3dis_best.pt")


if __name__ == "__main__":
    main()