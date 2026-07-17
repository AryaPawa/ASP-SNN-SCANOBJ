"""
eval_scanobj.py — Evaluate ASP-SNN on ScanObjectNN PB-T50-RS test set.

Supports test-time augmentation (TTA) with N-vote averaging, SSP-mode
override for ablation runs, and optional spike-rate + energy measurement.

Usage:
    python eval_scanobj.py --ckpt checkpoints/scanobj_best.pt
    python eval_scanobj.py --ckpt checkpoints/scanobj_best.pt --n_votes 10
    python eval_scanobj.py --ckpt ... --ssp_mode random --energy
"""

import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import load_config, set_seed
from datasets.scanobjectnn import ScanObjectNNDataset
from datasets.slicing import compute_geo_torch
from models.asp_classifier import ASPClassifier


def augment_vote_gpu(slices, geo):
    """
    One random z-rotation augmentation on GPU tensors for TTA.
    Recomputes full geo from rotated slices so SSP sees consistent geometry.
    """
    device, dtype = slices.device, slices.dtype
    theta = float(np.random.uniform(0, 2 * math.pi))
    c, s = math.cos(theta), math.sin(theta)
    rot = torch.tensor(
        [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]],
        device=device, dtype=dtype,
    )

    B, M, K, C = slices.shape
    slices_aug = slices.clone()
    slices_aug[:, :, :, :3] = slices[:, :, :, :3] @ rot
    if C >= 6:
        slices_aug[:, :, :, 3:6] = slices[:, :, :, 3:6] @ rot

    geo_aug = compute_geo_torch(slices_aug)
    return slices_aug, geo_aug


class _EncoderSpikeMonitor:
    """
    Hooks the spiking encoder's LIF layers (lif1, lif2_edge, lif3, lif4) and
    records their spike outputs to compute mean firing rate for energy report.
    Attach with .attach(spiking_encoder_impl), detach with .detach().
    """

    def __init__(self):
        self.total_spikes = 0
        self.total_neurons = 0
        self._handles = []

    def _make_hook(self):
        def hook(module, inp, out):
            self.total_spikes += out.sum().item()
            self.total_neurons += out.numel()
        return hook

    def attach(self, spiking_encoder_impl):
        """spiking_encoder_impl is a SpikingEdgeConvEncoder instance."""
        for name in ['lif1', 'lif2_edge', 'lif3', 'lif4']:
            layer = getattr(spiking_encoder_impl, name, None)
            if layer is not None:
                h = layer.register_forward_hook(self._make_hook())
                self._handles.append(h)

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def mean_rate(self):
        if self.total_neurons == 0:
            return 0.0
        return self.total_spikes / self.total_neurons

    def reset(self):
        self.total_spikes = 0
        self.total_neurons = 0


def evaluate(model, loader, device, n_votes=1, measure_energy=False):
    """Evaluate with optional TTA and optional spike-rate measurement."""
    model.eval()
    all_probs = []
    all_labels = []
    total_slices = 0
    total_samples = 0

    # Attach spike monitors: LIF head always; encoder only if spiking
    head_logger = None
    enc_monitor = None
    if measure_energy:
        from models.lif import SpikeRateLogger
        head_logger = SpikeRateLogger()
        model.lif_head.spike_monitor = head_logger

        # If encoder is spiking, attach hook-based monitor
        impl = getattr(model.feature_extractor, 'impl', None)
        if impl is not None and impl.__class__.__name__ == 'SpikingEdgeConvEncoder':
            enc_monitor = _EncoderSpikeMonitor()
            enc_monitor.attach(impl)

    with torch.no_grad():
        for slices, geo, labels in loader:
            slices = slices.to(device)
            geo = geo.to(device)
            B = slices.shape[0]

            summed = torch.zeros(B, model.cfg.num_classes, device=device)

            for v in range(n_votes):
                if v == 0:
                    s_v, g_v = slices, geo
                else:
                    s_v, g_v = augment_vote_gpu(slices, geo)

                logits_all = model(s_v, g_v, training=False)
                summed += logits_all[-1].softmax(dim=-1)

                if v == 0:
                    total_slices += len(logits_all) * B
                    total_samples += B

            all_probs.append((summed / n_votes).cpu())
            all_labels.append(labels)

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    preds = probs.argmax(dim=-1)

    oa = (preds == labels).float().mean().item()

    num_classes = model.cfg.num_classes
    per_class_correct = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)
    for c in range(num_classes):
        mask = labels == c
        per_class_total[c] = mask.sum().item()
        per_class_correct[c] = (preds[mask] == c).sum().item()

    per_class_acc = per_class_correct / per_class_total.clamp(min=1)
    macc = per_class_acc.mean().item()
    avg_slices = total_slices / max(total_samples, 1)

    head_fr = None
    encoder_fr = None
    if head_logger is not None:
        head_fr = head_logger.mean_rate()
        model.lif_head.spike_monitor = None
    if enc_monitor is not None:
        encoder_fr = enc_monitor.mean_rate()
        enc_monitor.detach()

    return oa, macc, per_class_acc, avg_slices, head_fr, encoder_fr


def main():
    p = argparse.ArgumentParser(description="Evaluate ScanObjectNN")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/scanobj_cls.yaml")
    p.add_argument("--n_votes", type=int, default=1,
                   help="Number of TTA votes (1=no TTA)")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--energy", action="store_true",
                   help="Measure spike rates and report energy accounting")
    p.add_argument("--ssp_mode", type=str, default=None,
                   choices=['learned', 'random', 'fps_order'],
                   help="Override SSP mode (for ablation eval)")
    p.add_argument("--set", nargs="*", default=[],
                   help="Override config values: --set key=value ...")
    args = p.parse_args()

    # Parse --set overrides (same format as training)
    extra = {}
    for item in args.set:
        if '=' in item:
            k, v = item.split('=', 1)
            # crude cast: try int, then float, then bool, else str
            try:
                extra[k] = int(v)
            except ValueError:
                try:
                    extra[k] = float(v)
                except ValueError:
                    if v.lower() == 'true':
                        extra[k] = True
                    elif v.lower() == 'false':
                        extra[k] = False
                    else:
                        extra[k] = v

    cfg = load_config(args.config, extra)
    if args.batch:
        cfg.batch_size = args.batch
    if args.ssp_mode:
        cfg.ssp_mode = args.ssp_mode
    set_seed(cfg.seed)
    device = cfg.device

    test_ds = ScanObjectNNDataset(cfg.data_dir, 'test', cfg)
    loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    cfg.in_channels = 6
    model = ASPClassifier(cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)

    print(f"\nCheckpoint : {args.ckpt}")
    print(f"Epoch      : {ckpt.get('epoch', '?')}")
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Encoder    : {getattr(cfg, 'encoder_type', 'analog')}"
          f"  (T_enc={getattr(cfg, 'encoder_T', 4)})")
    print(f"SSP mode   : {getattr(cfg, 'ssp_mode', 'learned')}")
    print(f"TTA votes  : {args.n_votes}")

    oa, macc, per_class_acc, avg_slices, head_fr, enc_fr = evaluate(
        model, loader, device, args.n_votes, measure_energy=args.energy,
    )

    print(f"\n{'='*50}")
    print(f"  Overall Accuracy  : {oa*100:.2f}%")
    print(f"  Mean Class Acc    : {macc*100:.2f}%")
    print(f"  Avg slices used   : {avg_slices:.2f} / {cfg.T}")
    print(f"{'='*50}")

    print(f"\n  Per-class accuracy ({cfg.num_classes} classes):")
    for c in range(cfg.num_classes):
        acc = per_class_acc[c].item()
        bar = "#" * int(acc * 30)
        print(f"    Class {c:2d}  {acc*100:5.1f}%  {bar}")

    if args.energy and head_fr is not None:
        from models.energy import compute_energy, print_energy_report
        energy = compute_energy(cfg, mean_firing_rate_head=head_fr,
                                mean_firing_rate_encoder=enc_fr)
        print_energy_report(energy)
    print()


if __name__ == "__main__":
    main()