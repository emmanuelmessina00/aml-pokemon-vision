import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class PokemonVisionDataset(Dataset):
    def __init__(self, num_samples=300, image_size=224, transform=None):
        self.num_samples = num_samples
        self.image_size = image_size


        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        img_np = np.uint8(np.random.rand(self.image_size, self.image_size, 3) * 255)
        img = Image.fromarray(img_np)


        label = torch.tensor(idx % 5, dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label