from pathlib import Path
import json
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# работаем со стандартными форматами изображений
EXTENSIONS = {'.jpg', '.jpeg', '.png'}

class ASLDataset(Dataset):
    def __init__(self, data_directory, path_to_split_json, set_name, transform=None):
        self.data_directory = Path(data_directory)

        with open(path_to_split_json, 'r', encoding='utf-8') as f:
            self.split = json.load(f)

        self.class_indices = self.split['class_indices']
        self.images = self.split[set_name]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.data_directory / self.images[index]
        image = Image.open(image_path).convert('RGB')
        class_name = Path(self.images[index]).parent.name
        target = self.class_indices[class_name]

        if self.transform is not None:
            image = self.transform(image)
        return image, target

def train_transform(image_size, mean, std):
    return transforms.Compose( [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
