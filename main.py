import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb

from dataset.pokemon_dataset import PokemonVisionDataset
from models.classifier import get_pokemon_resnet
from utils.stats import get_model_stats

if __name__ == "__main__":
    wandb.init(project="aml-pokemon-vision", name="test-pipeline-lab3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PokemonVisionDataset(num_samples=160)
    train_ds, val_ds, test_ds = random_split(dataset, [100, 30, 30])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    model = get_pokemon_resnet(num_classes=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    get_model_stats(train_loader, test_loader, val_loader, model, optimizer, criterion, device, 3, "ResNet18_Test")
    wandb.finish()