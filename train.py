import torch
import torch.nn as nn
from torchvision import models
import wandb
from dataset.pokemon_dataset import PokemonVisionDataset
from torch.utils.data import DataLoader, random_split
from train import train_loop
from eval import eval_loop
from utils.stats import get_model_stats

def train_loop(dataloader,model,optimizer,criterion,device):
  model.train()
  total_corrects=0
  train_loss=0
  total=0

  for x,y in dataloader:
    optimizer.zero_grad()
    x,y=x.to(device),y.to(device)
    Y_hat=model(x)
    loss= criterion(Y_hat,y)
    loss.backward()
    optimizer.step()

    train_loss +=loss.item() * x.size(0)
    total+=y.size(0)
    predicted_classes=Y_hat.argmax(dim=1)
    total_corrects+=(predicted_classes==y).sum().item()

  avg_loss=train_loss/total
  acc=total_corrects/total
  return avg_loss,acc

if __name__=="__main__":
    pass
