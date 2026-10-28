import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class PokemonVisionDataset(Dataset):
    def __init__(self, num_samples=500, image_size=224, transform=None):
        self.num_samples=num_samples
        self.image_size=image_size
        self.transform = transform

        print(f"[*] Inizializzato Dataset Pokémon con {num_samples} campioni.")

    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        # Generiamo un'immagine casuale fittizia (poi caricheremo i Pokémon)
        img = Image.fromarray(np.uint8(np.random.rand(self.image_size, self.image_size, 3) * 255))
        label = idx % 5  # Supponiamo  classi di Pokémon iniziali

        if self.transform:
            img = self.transform(img)

        return img, label

if __name__=="__main__":
    pass