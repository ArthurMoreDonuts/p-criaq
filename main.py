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
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    #entity="p-criaq",
    # Set the wandb project where this run will be logged.
    project="Testing-ComputeCanada",
    # Track hyperparameters and run metadata.
    config={
        
        "learning_rate": 0.0001,
        "architecture": "resnet34",
        "dataset": "SpeedDataset",
        "epochs": 25,
    },
)
epochs = 25
bsz = 32
lr = 0.0001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Hold the best model
#best_mse = np.inf   # init to infinity
#best_val_mse = np.inf
#best_val_weights = None
#best_weights = None
history = [] 
  # DATASET
#v2.Grayscale(3)
train_dataset = SpeedDataset(annotations_file=  '/../../../../../scratch/SpeedDataset/Labels.csv',
                                    img_dir= '/../../../../../scratch/SpeedDataset/train/original/frame',
                                     transform=transforms.Compose([
                                               
                                               
                                               v2.PILToTensor()])
                                          )
val_dataset = SpeedDataset(annotations_file=  '/../../../../../scratch/SpeedDataset/validation/ValLabels.csv',
                                    img_dir= '/../../../../../scratch/SpeedDataset/validation/original/frame',
                                           transform=transforms.Compose([
                                        
                                              
                                               v2.PILToTensor()]))
trn_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=bsz,
                        shuffle=False)

  # MODEL
model = timm.create_model('resnet34', pretrained = True, num_classes= 1)
#model = models.resnet50(pretrained=True)
#model.fc = nn.Linear(2048, 1)
#model = GaugeNet()
model.to(device)
oldLoss = sys.float_info.max
  #OPTIM
optimizer = optim.Adam(model.parameters(), lr=lr)
MSE = torch.nn.MSELoss()
#MSE = torch.nn.L1Loss()
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
    
    if loss.item() < best_mse:
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
model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
plt.plot(history)
plt.show()
    #if oldLoss > val_loss.item():
        #oldLoss = val_loss.item()
        #torch.save(model.state_dict(), 'models/Original64SpeedBestModel.pth')
    #else:
        #break

    
      # METRIC
      #max_pred = torch.argsort(pred, dim=1, descending=True)  
      #max_pred = max_pred[:,0]
      #print( f" For epoch {ep} Loss: {loss} 
      #hour_acc = float(torch.sum(max_h == hour)) / bsz
      #minute_acc = float(torch.sum(torch.abs(max_m - minute) <= 1)) / bsz

      #update_train_log(train_log, loss_cls, loss_reg, hour_acc, minute_acc)
      #if i == 0:
       # writer.add_images('train', img, ep)
    #write_train_log(writer, train_log, use_stn, ep)

    

    #torch.save(model.state_dict(), '../models/{}.pth'.format(verbose))  
   