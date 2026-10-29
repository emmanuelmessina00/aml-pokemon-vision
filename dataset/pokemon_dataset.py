import os
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from data.download_sprites import download_sprites

LABEL_MAP = {
    "background": 0,
    "bulbasaur": 1,
    "charmander": 2,
    "squirtle": 3,
    "pikachu": 4,
    "gengar": 5
}

CLASS_NAMES = list(LABEL_MAP.keys())


class PokemonVisionDataset(Dataset):
    def __init__(self, num_samples=300, image_size=224, sprite_dir='data/sprites', transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.sprite_dir = sprite_dir

        self.transform = transform or transforms.Compose([
            transforms.ToTensor()
        ])

        if not os.path.exists(sprite_dir) or len(os.listdir(sprite_dir)) == 0:
            download_sprites(sprite_dir)

        self.pokemon_names = [name for name in LABEL_MAP.keys() if name != "background"]
        self.label_map = LABEL_MAP

        self.sprites = {}
        for name in self.pokemon_names:
            path = os.path.join(sprite_dir, f"{name}.png")
            self.sprites[name] = Image.open(path).convert('RGBA')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Sfondo RGB casuale
        bg_color = np.random.randint(180, 255, size=(3,), dtype=np.uint8)
        canvas = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * bg_color
        bg_img = Image.fromarray(canvas, mode="RGB")

        # Scelta casuale del Pokémon ed estrazione label corretta da LABEL_MAP
        pokemon_name = random.choice(self.pokemon_names)
        label = self.label_map[pokemon_name]
        sprite = self.sprites[pokemon_name].copy()

        # Ridimensionamento casuale
        new_size = random.randint(70, 140)
        sprite = sprite.resize((new_size, new_size), Image.Resampling.BILINEAR)

        # Posizionamento casuale
        max_x = self.image_size - new_size
        max_y = self.image_size - new_size
        pos_x = random.randint(0, max_x)
        pos_y = random.randint(0, max_y)

        # Incolla sprite sul canvas
        bg_img.paste(sprite, (pos_x, pos_y), sprite)

        # Generazione maschera binaria dal canale Alpha
        alpha = np.array(sprite.split()[-1])
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        mask[pos_y:pos_y + new_size, pos_x:pos_x + new_size] = (alpha > 50).astype(np.uint8)

        # Calcolo coordinate bounding box: [x_min, y_min, x_max, y_max]
        non_zero_coords = np.argwhere(mask > 0)
        if len(non_zero_coords) > 0:
            y_min, x_min = non_zero_coords.min(axis=0)
            y_max, x_max = non_zero_coords.max(axis=0)
            bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
        else:
            bbox = torch.tensor([pos_x, pos_y, pos_x + new_size, pos_y + new_size], dtype=torch.float32)

        img_tensor = self.transform(bg_img)
        label_tensor = torch.tensor(label, dtype=torch.int64)

        target = {
            "boxes": bbox.unsqueeze(0),
            "labels": label_tensor.unsqueeze(0),
        }
        return img_tensor, target