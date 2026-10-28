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
    # Inizializzazione di WandB
    wandb.init(project="aml-pokemon-vision", config={
        "epochs": 15,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "architecture": "ResNet18"
    })


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Utilizzo device: {device}")


    dataset = PokemonVisionDataset(num_samples=600)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    eval_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)


    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model = model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()


    get_model_stats(
        train_loader=train_loader,
        test_loader=test_loader,
        eval_loader=eval_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        n_epochs=15,
        model_name="ResNet18_Pokemon"
    )

    wandb.finish()
