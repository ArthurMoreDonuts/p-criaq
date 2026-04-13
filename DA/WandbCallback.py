import wandb
from skorch.callbacks import Callback
import torch
import copy

class WandbCallback(Callback):
   
    def __init__(self, project_name="dan-training", config=None, save_path="best_model.pt"):
        self.project_name = project_name
        self.config = config or {}
        self.save_path = save_path
        self.best_valid_loss = float("inf")
        
    def on_train_begin(self, net, X=None, y=None, **kwargs):
        wandb.init(
            project=self.project_name,
            config={
                "lr": net.lr,
                "max_epochs": net.max_epochs,
                "batch_size": net.batch_size,
                **self.config
            }
        )

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        
        # Get the last entry in training history

        history = net.history[-1]
        
        log_dict = {"epoch": history["epoch"]}
        
        # Log train loss if available
        if "train_loss" in history:
            log_dict["train_loss"] = history["train_loss"]
        
        # Log valid loss if available (only if train_split is not None)
        if "valid_loss" in history:
            valid_loss = history["valid_loss"]
            if valid_loss <  self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.best_weights = copy.deepcopy(
                    net.module_.base_module_.model.state_dict()
                )
                print(f"  → New best weights stored in memory (valid_loss={valid_loss:.4f})")

            log_dict["valid_loss"] = valid_loss

        wandb.log(log_dict)

    def on_train_end(self, net, X=None, y=None, **kwargs):
        torch.save(
                    self.best_weights ,
                    self.save_path
                )
        wandb.finish()