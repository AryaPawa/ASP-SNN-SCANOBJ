"""
models/dgcnn_teacher.py — DGCNN classifier for use as a Knowledge Distillation teacher.

Standard DGCNN architecture from Wang et al. "Dynamic Graph CNN for Learning on
Point Clouds" (SIGGRAPH 2019). Used ONLY to train a teacher for KD — the
teacher is never active at ASP-SNN inference, so it has zero impact on the
efficiency claim (system α, per-sample energy).

Reference architecture — the vanilla DGCNN classification network:
    - 4 stacked EdgeConv layers with k=20 neighbors
    - Feature dims: 64 -> 64 -> 128 -> 256, all concatenated to 512 channels
    - MLP fusion: 1024 -> 512 -> 256 -> num_classes
    - Global max pooling for point-set aggregation

We deliberately keep this ARCHITECTURALLY SEPARATE from ASP-SNN's EdgeConv
encoder so the teacher provides genuinely different signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_dgcnn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Find k nearest neighbors in feature space (dynamic graph).
    x: [B, C, N] -> idx: [B, N, k] indices.
    """
    B, C, N = x.shape
    # Pairwise squared distances
    inner = -2 * torch.matmul(x.transpose(2, 1), x)              # [B, N, N]
    xx = (x ** 2).sum(dim=1, keepdim=True)                        # [B, 1, N]
    dist = -xx - inner - xx.transpose(2, 1)                       # [B, N, N]
    idx = dist.topk(k=k, dim=-1)[1]                               # [B, N, k]
    return idx


def get_graph_features(x: torch.Tensor, k: int = 20,
                       idx: torch.Tensor = None) -> torch.Tensor:
    """
    Build edge features for EdgeConv.
    x: [B, C, N] -> feats: [B, 2C, N, k]
    """
    B, C, N = x.shape
    if idx is None:
        idx = knn_dgcnn(x, k)                                     # [B, N, k]

    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = idx + idx_base                                           # [B, N, k]
    idx = idx.view(-1)                                             # [B*N*k]

    x = x.transpose(2, 1).contiguous()                             # [B, N, C]
    feature = x.view(B * N, -1)[idx, :]                            # [B*N*k, C]
    feature = feature.view(B, N, k, C)                             # [B, N, k, C]

    x = x.view(B, N, 1, C).expand(-1, -1, k, -1)                   # [B, N, k, C]
    # Concatenate [neighbor - center, center] as the standard DGCNN edge feature
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature                                                 # [B, 2C, N, k]


class DGCNNTeacher(nn.Module):
    """
    Vanilla DGCNN point-cloud classifier.

    Input:  points [B, N, 3]  (xyz only — this is the standard DGCNN input)
    Output: logits [B, num_classes]
    """

    def __init__(self, num_classes: int = 15, k: int = 20,
                 emb_dims: int = 1024, dropout: float = 0.5):
        super().__init__()
        self.k = k

        # 4 EdgeConv layers — with LeakyReLU as per original DGCNN
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3 * 2, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        # Fuse concatenated features
        self.conv5 = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )

        # Classification head
        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, 3] point cloud (xyz only).
        Returns logits: [B, num_classes].
        """
        # DGCNN uses [B, 3, N] convention internally
        x = x.transpose(2, 1).contiguous()                        # [B, 3, N]
        B = x.size(0)

        x1 = get_graph_features(x, k=self.k)                      # [B, 6, N, k]
        x1 = self.conv1(x1)                                       # [B, 64, N, k]
        x1 = x1.max(dim=-1, keepdim=False)[0]                     # [B, 64, N]

        x2 = get_graph_features(x1, k=self.k)                     # [B, 128, N, k]
        x2 = self.conv2(x2)                                       # [B, 64, N, k]
        x2 = x2.max(dim=-1, keepdim=False)[0]                     # [B, 64, N]

        x3 = get_graph_features(x2, k=self.k)                     # [B, 128, N, k]
        x3 = self.conv3(x3)                                       # [B, 128, N, k]
        x3 = x3.max(dim=-1, keepdim=False)[0]                     # [B, 128, N]

        x4 = get_graph_features(x3, k=self.k)                     # [B, 256, N, k]
        x4 = self.conv4(x4)                                       # [B, 256, N, k]
        x4 = x4.max(dim=-1, keepdim=False)[0]                     # [B, 256, N]

        x = torch.cat((x1, x2, x3, x4), dim=1)                    # [B, 512, N]
        x = self.conv5(x)                                          # [B, emb_dims, N]

        # Global pooling
        x1 = F.adaptive_max_pool1d(x, 1).view(B, -1)              # [B, emb_dims]
        x2 = F.adaptive_avg_pool1d(x, 1).view(B, -1)              # [B, emb_dims]
        x = torch.cat((x1, x2), 1)                                # [B, emb_dims*2]

        # MLP
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)                                       # [B, num_classes]
        return x