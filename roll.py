import os
import os.path
import time
import torch
import sys
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torchvision.transforms import v2, transforms
import copy
import tqdm
import random
import timm
from RollDataset import RollDataset
import wandb
# Start a new wandb run to track this script.




epochs = 100
bsz = 32
lr = 0.0001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Hold the best model

history = [] 
  # DATASET
#v2.Grayscale(3)
train_dataset = RollDataset(annotations_file=  '/home/o7ahmed/scratch/Roll&PitchDataset/Labels.csv',
                                    img_dir= '/home/o7ahmed/scratch/Roll&PitchDataset/train/frame',
                                     transform=transforms.Compose([
                                               
                                               v2.Resize((224,224)),
                                               v2.PILToTensor()])
                                          )
val_dataset = RollDataset(annotations_file=  '/home/o7ahmed/scratch/Roll&PitchDataset/valLabels.csv',
                                    img_dir= '/home/o7ahmed/scratch/Roll&PitchDataset/validation/frame',
                                           transform=transforms.Compose([
                                        
                                              v2.Resize((224,224)),
                                               v2.PILToTensor()]))

trn_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=bsz,
                        shuffle=False)
#models = [ 'efficientnet_b0.ra_in1k','efficientvit_b0.r224_in1k', 'nextvit_small.bd_in1k','convnext_small.in12k', 'convnextv2_base.fcmae_ft_in1k',  'convit_base.fb_in1k',    'resnetv2_50x1_bit.goog_distilled_in1k',    'beit_base_patch16_224.in22k_ft_in22k',  'cspresnet50.ra_in1k', 'deit3_base_patch16_224.fb_in1k', 'densenet121.ra_in1k', 'dla34.in1k',  'edgenext_base.in21k_ft_in1k','efficientformer_l1.snap_dist_in1k','efficientformerv2_l.snap_dist_in1k', 'fastvit_ma36.apple_dist_in1k',  'gcvit_base.in1k', 'ghostnet_100.in1k','ghostnetv2_100.in1k', 'gmlp_s16_224.ra3_in1k','hgnet_base.ssld_in1k','hrnet_w18.ms_aug_in1k','inception_next_base.sail_in1k','lambda_resnet26t.c1_in1k', 'levit_256.fb_dist_in1k','mobilenetv1_100.ra4_e3600_r224_in1k','mobilenetv2_120d.ra_in1k','mobilenetv3_small_075.lamb_in1k','mobilenetv4_conv_medium.e500_r224_in1k','mobileone_s2.apple_in1k','mobilevit_s.cvnets_in1k','mobilevitv2_200.cvnets_in1k','nasnetalarge.tf_in1k','nest_base_jx.goog_in1k','nf_resnet50.ra2_in1k','nf_regnet_b1.ra2_in1k','nfnet_l0.ra2_in1k','pit_b_224.in1k','pnasnet5large.tf_in1k','poolformer_m36.sail_in1k','pvt_v2_b0.in1k','rdnet_base.nv_in1k','regnetv_040.ra3_in1k','repghostnet_050.in1k','repvgg_a0.rvgg_in1k','repvit_m0_9.dist_300e_in1k','resmlp_12_224.fb_distilled_in1k','sam2_hiera_base_plus.fb_r896','samvit_base_patch16.sa1b','selecsls42b.in1k','seresnet50.a1_in1k','twins_pcpvt_base.in1k','visformer_small.in1k','vit_base_patch8_224.augreg2_in21k_ft_in1k','vit_base_patch32_clip_quickgelu_224.metaclip_400m','vitamin_small_224.datacomp1b_clip', 'xcit_small_24_p16_224.fb_in1k','skresnet18.ra_in1k','swin_base_patch4_window7_224.ms_in1k','tf_efficientnet_b0.aa_in1k','tiny_vit_5m_224.in1k',]


models = [  'coat_small.in1k' ,
            'deit_base_patch16_224.fb_in1k',
            'convnext_base.fb_in22k',
            'dpn68.mx_in1k',
            'focalnet_base_lrf.ms_in1k',
            'mambaout_base.in1k',
            'maxvit_base_tf_224.in1k',
            'mvitv2_base.fb_in1k',
            'nextvit_base.bd_in1k',
            'tresnet_l.miil_in1k',
            'volo_d1_224.sail_in1k',
            'xception41.tf_in1k',
            ]

# Start a new wandb run to track this script.

for x in models:
    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    #entity="p-criaq",
    # Set the wandb project where this run will be logged.
    project="Nibi",
    # Track hyperparameters and run metadata.
    config={
        
        "learning_rate": lr,
        "architecture": x,
        "dataset": "RollDataset - With pitch ",
        "epochs": epochs,
    },
    )
    model = timm.create_model(x, pretrained = True, num_classes= 1)
    model.to(device)
    oldLoss = sys.float_info.max
    #OPTIM
    optimizer = optim.Adam(model.parameters(), lr=lr)
    MSE = torch.nn.MSELoss()
    #MSE = torch.nn.L1Loss()
    best_mse = np.inf   # init to infinity
    best_val_mse = np.inf
    best_val_weights = None
    best_weights = None
    for ep in range(epochs):
    
        cLoss = 0    
        #for i, trn_sample in enumerate(tqdm.tqdm(trn_loader)):
        for i, (img , speed) in enumerate(trn_loader):
          model.train()
          optimizer.zero_grad()
        
          img = img/255
          img = img.to(device)
          speed = speed.float().to(device)


          # PREDICT
      
          pred = model(img)
          loss = MSE(pred.flatten(),speed)
          cLoss += loss.item()

          # LOSS
      
          #optimizer.zero_grad()
          loss.backward()
          optimizer.step()
    
        val_loss = 0
        model.eval()
    
        if cLoss < best_mse:
            best_mse = loss.item()
            best_weights = copy.deepcopy(model.state_dict())
        for i, (img, speed) in enumerate(val_loader):
      # Validation step
            img = img/255
            img = img.to(device)
            speed = speed.float().to(device)
        
      
            with torch.no_grad():
                val_outputs = model(img)
                val_loss += MSE(val_outputs.flatten(), speed).item()
        run.log({ "training loss": cLoss, "Validation loss": val_loss})
        if val_loss < best_val_mse:
            best_val_mse = val_loss
            best_val_weights = copy.deepcopy(model.state_dict())
        history.append(val_loss)
            #val_acc = accuracy_score(speed.cpu(), val_outputs.cpu())
    
        print(f'Epoch [{ep+1}/{epochs}], Loss: {cLoss:.4f}, Val Loss: {val_loss:.4f}')
    
    

    # restore model and return best accuracy
    model.load_state_dict(best_val_weights)
    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))
    torch.save(model.state_dict(), '/home/o7ahmed/scratch/RollModels/'+x+'RollBestModel.pth')
    run.finish()
    
   