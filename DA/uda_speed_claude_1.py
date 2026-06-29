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


def gaussian_kernel(x, y, bandwidth):
    """Compute Gaussian (RBF) kernel between two feature matrices."""
    # x, y: (N, D)
    xx = (x ** 2).sum(1, keepdim=True)          # (N, 1)
    yy = (y ** 2).sum(1, keepdim=True)          # (M, 1)
    xy = x @ y.T                                 # (N, M)
    sq_dist = xx + yy.T - 2 * xy                 # (N, M) pairwise squared distances
    return torch.exp(-sq_dist / (2 * bandwidth ** 2))


def mkmmd_loss(source_features, target_features, bandwidths=None):
    """
    Multiple Kernel MMD between source and target feature batches.

    Args:
        source_features: (N, D) source batch features
        target_features: (M, D) target batch features  
        bandwidths: list of RBF bandwidth values (default: [0.5, 1, 2, 4, 8])

    Returns:
        Scalar MK-MMD loss
    """
    if bandwidths is None:
        bandwidths = [0.5, 1.0, 2.0, 4.0, 8.0]

    loss = torch.tensor(0.0, device=source_features.device)

    for bw in bandwidths:
        K_ss = gaussian_kernel(source_features, source_features, bw)   # (N, N)
        K_tt = gaussian_kernel(target_features, target_features, bw)   # (M, M)
        K_st = gaussian_kernel(source_features, target_features, bw)   # (N, M)

        # MMD^2 = E[K_ss] - 2*E[K_st] + E[K_tt]
        mmd2 = K_ss.mean() - 2 * K_st.mean() + K_tt.mean()
        loss = loss + mmd2

    return loss / len(bandwidths)

class ConvNextUDA(nn.Module):
    def __init__(self, model_name="convnext_small.in12k_ft_in1k", num_outputs=1):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs)
        )

    def forward(self, x):
        features = self.backbone(x)
        preds    = self.regressor(features)
        return features, preds


def get_lambda(current_step, total_steps):
    """Ramp λ from ~0 to 1 over training (avoids MMD destabilising early regression)."""
    progress = current_step / total_steps
    return 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * progress)).item()) - 1.0


def finetune_uda(model, source_loader, target_loader,
                 optimizer, num_epochs=20,
                 freeze_backbone=False, lambda_max=1.0):
    """
    Load source-pretrained weights, then adapt to target domain via MK-MMD.

    freeze_backbone=True  → only the regressor head learns; backbone is a fixed
                            feature extractor. Faster and more stable when the
                            domain gap is small.
    freeze_backbone=False → entire network fine-tunes. Use when the domain gap
                            is large enough to require backbone adaptation.
    """
    # Load the source-pretrained checkpoint
    #model.load_state_dict(torch.load("model\\convnext_small.in12kSpeedBestModel.pth"))
    #model = model.cuda()
    #print("Loaded source-pretrained weights for UDA fine-tuning.")

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen — only regression head will be updated.")
    else:
        print("Full network will be fine-tuned.")

    model.train()
    criterion = nn.MSELoss()
    total_steps = num_epochs * min(len(source_loader), len(target_loader))
    step = 0

    for epoch in range(num_epochs):
        for (src_imgs, src_labels), (tgt_imgs, _) in zip(source_loader, target_loader):
            src_imgs   = src_imgs.cuda().float()
            src_labels = src_labels.cuda().float()
            tgt_imgs   = tgt_imgs.cuda().float()

            src_feats, src_preds = model(src_imgs)
            tgt_feats, _         = model(tgt_imgs)

            loss_reg = criterion(src_preds.squeeze(), src_labels)
            loss_mmd = mkmmd_loss(src_feats, tgt_feats)

            lam  = get_lambda(step, total_steps) * lambda_max
            loss = loss_reg + lam * loss_mmd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

        print(f"[UDA] Epoch {epoch+1}/{num_epochs}  "
              f"loss_reg={loss_reg.item():.4f}  "
              f"loss_mmd={loss_mmd.item():.4f}  "
              f"λ={lam:.3f}")


def load_pretrained_convnext(checkpoint_path, model_name="convnext_small.in12k", num_outputs=1):
    """
    Load a previously trained ConvNext into the UDA wrapper, handling the
    most common checkpoint formats.
    """
    uda_model = ConvNextUDA(model_name=model_name, num_outputs=num_outputs).cuda()
    checkpoint = torch.load(checkpoint_path, map_location="cuda")

    # Unwrap common checkpoint wrappers
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state_dict") or
            checkpoint.get("state_dict") or
            checkpoint.get("model") or
            checkpoint          # assume it IS the state dict already
        )
    else:
        # torch.save(model) was used — pull state dict from the full model object
        state_dict = checkpoint.state_dict()

    # ── Remap keys to match ConvNextUDA ──────────────────────────────────────
    # Case 1: keys already have "backbone." prefix → load directly
    # Case 2: plain timm keys (no prefix, has "head." layers) → map to backbone.*
    # Case 3: keys have some other prefix (e.g. "module." from DataParallel) → strip it

    remapped = {}
    sample_key = next(iter(state_dict))

    if sample_key.startswith("backbone."):
        # Already in the right format
        remapped = state_dict

    elif sample_key.startswith("module."):
        # DataParallel wrapper — strip "module." prefix, then re-check
        stripped = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        sample_key2 = next(iter(stripped))
        if sample_key2.startswith("backbone."):
            remapped = stripped
        else:
            # Still a raw timm model after stripping DataParallel
            for k, v in stripped.items():
                if not k.startswith("head."):   # skip the original classifier
                    remapped[f"backbone.{k}"] = v

    else:
        # Raw timm model — map everything except the classifier head
        for k, v in state_dict.items():
            if not k.startswith("head."):
                remapped[f"backbone.{k}"] = v

    missing, unexpected = uda_model.load_state_dict(remapped, strict=False)

    # Expected: regressor keys will be missing (randomly initialised) — that's fine
    # Unexpected: warns you about anything that didn't map correctly
    backbone_missing    = [k for k in missing    if k.startswith("backbone.")]
    backbone_unexpected = [k for k in unexpected if k.startswith("backbone.")]

    if backbone_missing:
        print(f"WARNING — backbone keys not loaded ({len(backbone_missing)}): {backbone_missing[:5]}")
    if backbone_unexpected:
        print(f"WARNING — unexpected backbone keys ({len(backbone_unexpected)}): {backbone_unexpected[:5]}")

    regressor_missing = [k for k in missing if k.startswith("regressor.")]
    if regressor_missing:
        print(f"Regressor will train from scratch (expected): {regressor_missing}")

    print("Backbone loaded successfully.")
    return uda_model





epochs = 300
batch_size = 32
lr = 0.0001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
target = 1
# Hold the best model

history = [] 
  # DATASET
#v2.Grayscale(3)
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



#target_subset_dataset = Subset(target_dataset,list(range(3330,4556)))
#target_subset_dataset = Subset(target_dataset,list(range(1902,3330)))
#target_subset_dataset = Subset(target_dataset,list(range(1037,1902)))
#target_subset_dataset = Subset(target_dataset,list(range(621,1037)))
target_subset_dataset = Subset(target_dataset,list(range(0,621)))


#trn_loader = DataLoader(source_subset_dataset, batch_size=bsz, shuffle=True)
val_loader = DataLoader(target_subset_dataset, batch_size=batch_size, shuffle=True)




full_trn_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
#full_val_loader = DataLoader(target_dataset, batch_size=bsz, shuffle=False)



models = ["convnext_small.in12k",]
MSE = torch.nn.MSELoss()
for x in models:
    model = load_pretrained_convnext(
    checkpoint_path=f"/home/o7ahmed/scratch/SpeedModels/{x}SpeedBestModel.pth",
    model_name="convnext_small.in12k",  # must match your original architecture
    num_outputs=1
)

optimizer_phase2 = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5,
    weight_decay=1e-4
)
target_subset_dataset = Subset(target_dataset,list(range(0,621)))

source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
target_loader = DataLoader(target_subset_dataset, batch_size=batch_size,
                        shuffle=True)
# Skip pretrain_source entirely — your model is already pretrained
finetune_uda(model, source_loader, target_loader,
             optimizer_phase2, num_epochs=epochs,
             freeze_backbone=False,
             lambda_max=1.0)


torch.save({
        "model_state_dict": model.state_dict(),
    }, "/home/o7ahmed/scratch/SpeedModels/claude_speedmodel.pth")