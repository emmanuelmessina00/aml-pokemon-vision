import torch
from torch.utils.data import DataLoader
import wandb

from dataset.pokemon_dataset import PokemonVisionDataset
from models.detector import get_pokemon_detector
from utils.plots import log_prediction_samples
from utils.stats import collate_fn
from train import train_one_epoch_detection
from eval import evaluate_detection


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Utilizzo dispositivo: {device}")

    wandb.init(
        project="aml-pokemon-vision",
        name="faster-rcnn-pokemon-detection",
        config={
            "architecture": "Faster-RCNN-ResNet50-FPN",
            "epochs": 5,
            "batch_size": 4,
            "lr": 0.0005
        }
    )

    train_dataset = PokemonVisionDataset(num_samples=200)
    val_dataset = PokemonVisionDataset(num_samples=40)
    test_dataset = PokemonVisionDataset(num_samples=20)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    model = get_pokemon_detector(num_classes=6)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)

    print("\n[*] Inizio addestramento Object Detection...")
    epochs = 5
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_detection(model, optimizer, train_loader, device, epoch)
        val_loss = evaluate_detection(model, val_loader, device)

        print(f"Epoca [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
    print("[*] Generazione campioni visivi per WandB...")
    log_prediction_samples(model, test_dataset, device, num_samples=4, score_thresh=0.7)
    wandb.finish()
    print("[*] Addestramento completato con successo!")

if __name__ == "__main__":
    main()