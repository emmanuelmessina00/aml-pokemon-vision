import os
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from data.download_sprites import download_sprites


class PokemonVisionDataset(Dataset):
    def __init__(self, num_samples=300, image_size=224, sprite_dir='data/sprites',transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.sprite_dir = sprite_dir

        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if not os.path.exists(sprite_dir) or len(os.listdir(sprite_dir)) == 0:
            download_sprites(sprite_dir)

        self.class_names= sorted([f.replace('.png', '') for f in os.listdir(sprite_dir) if f.endswith('.png')])
        self.label_map= {name: idx for idx, name in enumerate(self.class_names)}

        self.sprites= {}
        for name in self.class_names:
            path = os.path.join(sprite_dir, f"{name}.png")
            self.sprites[name] = Image.open(path).convert('RGBA')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Sfondo RGB casuale (rumore uniforme morbido)
        bg_color = np.random.randint(180, 255, size=(3,), dtype=np.uint8)
        canvas = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * bg_color
        bg_img = Image.fromarray(canvas, mode="RGB")

        # Scelta Pokémon e classe
        class_name = random.choice(self.class_names)
        label = self.label_map[class_name]
        sprite = self.sprites[class_name].copy()

        # Ridimensionamento casuale dello sprite (tra 60px e 140px)
        new_size = random.randint(70, 140)
        sprite = sprite.resize((new_size, new_size), Image.Resampling.BILINEAR)

        # Posizionamento casuale sulla tela
        max_x = self.image_size - new_size
        max_y = self.image_size - new_size
        pos_x = random.randint(0, max_x)
        pos_y = random.randint(0, max_y)

        # Incolla usando il canale Alpha dello sprite come maschera
        bg_img.paste(sprite, (pos_x, pos_y), sprite)

        # Generazione Bounding Box e Maschera di Segmentazione
        alpha = np.array(sprite.split()[-1])  # Canale alpha dello sprite
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        mask[pos_y:pos_y + new_size, pos_x:pos_x + new_size] = (alpha > 50).astype(np.uint8)

        # Calcolo bounding box: [x_min, y_min, x_max, y_max]
        non_zero_coords = np.argwhere(mask > 0)
        if len(non_zero_coords) > 0:
            y_min, x_min = non_zero_coords.min(axis=0)
            y_max, x_max = non_zero_coords.max(axis=0)
            bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
        else:
            bbox = torch.tensor([pos_x, pos_y, pos_x + new_size, pos_y + new_size], dtype=torch.float32)

        # Applicazione trasformazioni all'immagine
        img_tensor = self.transform(bg_img)
        mask_tensor = torch.tensor(mask, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Restituiamo una struttura target adatta anche ai task standard di Torchvision
        target = {
            "boxes": bbox.unsqueeze(0),
            "labels": label_tensor.unsqueeze(0),
        }
        return img_tensor, target

