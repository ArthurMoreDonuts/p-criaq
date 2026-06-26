import os

import numpy as np
import pandas as pd
import os.path as osp
import torch
#import RRDBNet_arch as arch
import csv
from torchvision.io import decode_image
from torchvision.transforms import v2, transforms
from torch.utils.data import Dataset
import sys
import random
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch import nn

import copy
import timm
import wandb

from ConvNextWrapper import ConvNextWrapper
from SpeedDataset import SpeedDataset
from WandbCallback import WandbCallback
import torch
import torch.nn as nn


import torch
import torch.nn as nn

from torch.utils.data import DataLoader


from torch.utils.data import DataLoader, Subset
source_dataset = SpeedDataset(annotations_file=  '/home/o7ahmed/scratch/SpeedDataset/Labels.csv',
                                    img_dir= '/home/o7ahmed/scratch/SpeedDataset/train/original/frame',
                                     transform=transforms.Compose([
                                               
                                               v2.Resize((224,224)),
                                               v2.PILToTensor()])
                                          )
target_dataset = SpeedDataset(annotations_file='/home/o7ahmed/scratch/CrossDatasets/SpeedDataset/Labels.csv',
                                    img_dir='/home/o7ahmed/scratch/CrossDatasets/SpeedDataset/frame',
                                           transform=transforms.Compose([
                                        
                                               v2.Resize((224,224)),
                                               v2.PILToTensor()]))



target_subset_dataset = Subset(target_dataset,list(range(0,621)))

source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_loader = DataLoader(target_subset_dataset, batch_size=32,
                        shuffle=True)
# ── Helpers ────────────────────────────────────────────────────────────────────

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    total = torch.cat([source, target], dim=0)
    n = total.size(0)
    total0 = total.unsqueeze(0).expand(n, n, total.size(1))
    total1 = total.unsqueeze(1).expand(n, n, total.size(1))
    L2_dist = ((total0 - total1) ** 2).sum(dim=2)

    bandwidth = fix_sigma if fix_sigma else torch.sum(L2_dist.detach()) / (n**2 - n)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    return sum(torch.exp(-L2_dist / bw) for bw in bandwidth_list)


def mkmmd_loss(source_feat, target_feat, kernel_mul=2.0, kernel_num=5):
    bs = min(source_feat.size(0), target_feat.size(0))
    source_feat = source_feat[:bs]
    target_feat = target_feat[:bs]
    K = gaussian_kernel(source_feat, target_feat, kernel_mul, kernel_num)
    return torch.mean(K[:bs, :bs] + K[bs:, bs:] - K[:bs, bs:] - K[bs:, :bs])


def lambda_schedule(epoch, total_epochs, gamma=10.0):
    """Ramps lambda from ~0 → 1 following the DANN schedule."""
    progress = epoch / total_epochs
    return 2.0 / (1.0 + np.exp(-gamma * progress)) - 1.0


def infinite_loader(loader):
    """Cycles a DataLoader indefinitely."""
    while True:
        for batch in loader:
            yield batch

# ── Model ──────────────────────────────────────────────────────────────────────

class ConvNextUDA(nn.Module):
    def __init__(self, backbone_name='convnext_small.in12k_ft_in1k', pretrained_path=None, num_outputs=1):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location='cpu')
            state = ckpt.get('model_state_dict', ckpt)
            # strip DataParallel 'module.' prefix if present
            state = {k.replace('module.', ''): v for k, v in state.items()}
            self.backbone.load_state_dict(state, strict=False)

        feat_dim = self.backbone.num_features
        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs)
        )

    def forward(self, x):
        feat = self.backbone(x)  # (B, feat_dim)
        pred = self.regressor(feat)
        return pred, feat

# ── Main training function ─────────────────────────────────────────────────────

def train_uda(
    model,
    source_loader: DataLoader,
    target_loader: DataLoader,
    num_epochs: int = 50,
    lambda_mmd_max: float = 1.0,
    device: str = 'cuda',
    save_path: str = '/home/o7ahmed/scratch/SpeedModels/claude_best_model.pth',
    freeze_backbone_epochs: int = 5,      # warm-up: train head only
):
    model.to(device)
    task_criterion = nn.SmoothL1Loss()

    # Differential LRs: backbone gets 10× lower LR than head
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.regressor.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    target_iter = infinite_loader(target_loader)
    best_loss = float('inf')

    # ── Backbone freeze for warm-up ────────────────────────────────────────────
    def set_backbone_grad(requires_grad: bool):
        for p in model.backbone.parameters():
            p.requires_grad = requires_grad

    set_backbone_grad(False)   # frozen at start

    for epoch in range(num_epochs):

        # Unfreeze backbone after warm-up period
        if epoch == freeze_backbone_epochs:
            print(f"[Epoch {epoch+1}] Unfreezing backbone.")
            set_backbone_grad(True)

        lam = lambda_mmd_max * lambda_schedule(epoch, num_epochs)

        model.train()
        epoch_task_loss = 0.0
        epoch_mmd_loss  = 0.0
        epoch_total_loss = 0.0
        num_batches = 0

        for src_imgs, src_labels in source_loader:
            tgt_imgs = next(target_iter)
            # handle (img, label) or img-only target batches
            tgt_imgs = tgt_imgs[0] if isinstance(tgt_imgs, (list, tuple)) else tgt_imgs

            src_imgs   = src_imgs.float().to(device)
            src_labels = src_labels.to(device).float()
            tgt_imgs   = tgt_imgs.float().to(device)

            src_preds, src_feats = model(src_imgs)
            _,         tgt_feats = model(tgt_imgs)

            task_loss = task_criterion(src_preds.squeeze(-1), src_labels)
            mmd       = mkmmd_loss(src_feats, tgt_feats)
            loss      = task_loss + lam * mmd

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_task_loss  += task_loss.item()
            epoch_mmd_loss   += mmd.item()
            epoch_total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        avg_task = epoch_task_loss  / num_batches
        avg_mmd  = epoch_mmd_loss   / num_batches
        avg_tot  = epoch_total_loss / num_batches

        print(
            f"Epoch [{epoch+1:03d}/{num_epochs}] "
            f"λ={lam:.3f}  "
            f"task={avg_task:.4f}  "
            f"mmd={avg_mmd:.4f}  "
            f"total={avg_tot:.4f}"
        )

        # Save best checkpoint on task loss (source supervised signal)
        if avg_task < best_loss:
            best_loss = avg_task
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'task_loss': avg_task,
                'mmd_loss': avg_mmd,
            }, save_path)
            print(f"  ✓ Saved best model (task_loss={best_loss:.4f})")

    print("Training complete.")
    return model

# ── Entry point ────────────────────────────────────────────────────────────────
from torchvision import transforms, datasets

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


model = ConvNextUDA(
        backbone_name='convnext_small.in12k_ft_in1k',
        pretrained_path="/home/o7ahmed/scratch/SpeedModels/convnext_small.in12kSpeedBestModel.pth",
        num_outputs=1,
    )

train_uda(
        model=model,
        source_loader=source_loader,
        target_loader=target_loader,
        num_epochs=200,
        lambda_mmd_max=1.0,
        device='cuda',
        freeze_backbone_epochs=5,
    )
