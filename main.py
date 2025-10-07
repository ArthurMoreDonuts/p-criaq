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
from SpeedDataset import SpeedDataset
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
train_dataset = SpeedDataset(annotations_file=  '/home/o7ahmed/scratch/SpeedDataset/Labels.csv',
                                    img_dir= '/home/o7ahmed/scratch/SpeedDataset/train/original/frame',
                                     transform=transforms.Compose([
                                               
                                               
                                               v2.PILToTensor()])
                                          )
val_dataset = SpeedDataset(annotations_file=  '/home/o7ahmed/scratch/SpeedDataset/validation/ValLabels.csv',
                                    img_dir= '/home/o7ahmed/scratch/SpeedDataset/validation/original/frame',
                                           transform=transforms.Compose([
                                        
                                              
                                               v2.PILToTensor()]))

trn_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=bsz,
                        shuffle=False)
models = [ 'efficientnet_b0.ra_in1k','efficientvit_b0.r224_in1k', 'nextvit_small.bd_in1k']
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
        "dataset": "SpeedDataset",
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
    torch.save(model.state_dict(), '/home/o7ahmed/scratch/'+x+'SpeedBestModel.pth')
    run.finish()
    
   