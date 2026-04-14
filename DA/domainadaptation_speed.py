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
from skada.deep import DeepCoral
import copy
import timm
import wandb
from skada.deep import DAN
from skorch.callbacks import GradientNormClipping
from ConvNextWrapper import ConvNextWrapper
from SpeedDataset import SpeedDataset
from WandbCallback import WandbCallback
from skada.datasets import DomainAwareDataset

epochs = 10
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
    model = timm.create_model(x, pretrained = True, num_classes= 1)
    model.load_state_dict(torch.load(f"/home/o7ahmed/scratch/SpeedModels/{x}SpeedBestModel.pth", weights_only=True))         # your trained ConvNeXt
model.eval()

wrapper = ConvNextWrapper(model=model.cpu())

@torch.no_grad()
def extract_features(dataloader):
    feats, ys = [], []
    for x, y in dataloader: # for target, y = None or dummy
        x= x.float()/255
        y= y.float()
        feats.append(x)
        if y is not None:
            ys.append(y)
    X = torch.cat(feats).numpy()
    y = None if not ys else torch.cat(ys).numpy()
    return X, y

class MSELossFlat(nn.MSELoss):
    def forward(self, input, target):
        return super().forward(input, target.view_as(input))  # reshape target to match input


coralModel = DeepCoral(
    wrapper,
    layer_name="model.norm_pre",
    device=device,
    batch_size=batch_size,
    max_epochs=epochs,
    train_split=False,
    reg=0.001,
    lr=1e-5,
    base_criterion=MSELossFlat(),
    callbacks=[
        GradientNormClipping(gradient_clip_value=1.0),
        WandbCallback(
            project_name="Narval-DA",
            config={"trained on (1-5)": target, "method": "Deepcoral"},
            save_path=f"/home/o7ahmed/scratch/DAmodels/{x}_trained_on_{target}_Deepcoral_SpeedBestModel.pth"
        ),
    ],
)
wrapper = ConvNextWrapper(model=model.cpu())

danModel = DAN(
    module=wrapper,
    layer_name="model.norm_pre",
    device=device,
    lr=1e-5,
    max_epochs=epochs,
    batch_size=batch_size,
    train_split=None,
    base_criterion=MSELossFlat(),
    callbacks=[
        GradientNormClipping(gradient_clip_value=1.0),
        WandbCallback(
            project_name="Narval-DA",
            config={"trained on (1-5)": target, "method": "DAN"},
            save_path=f"/home/o7ahmed/scratch/DAmodels/{x}_trained_on_{target}_Dan_SpeedBestModel.pth"
        ),
    ],
)

#X_s, Y_s = extract_features(trn_loader)
X_s, Y_s = extract_features(full_trn_loader)
X_t, Y_t = extract_features(val_loader)


datasets = DomainAwareDataset()
datasets.add_domain(X_s, Y_s , domain_name="source")
datasets.add_domain(X_t, Y_t  , domain_name="target")

X, y, sample_domain = datasets.pack(as_sources=['source'], as_targets=['target'], mask_target_labels=True)

danModel.fit(X, y, sample_domain=sample_domain)
coralModel.fit(X, y, sample_domain=sample_domain)

