import os
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path):
        abs_path = os.path.abspath(folder_path)

        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Error: The folder path '{abs_path}' does not exist. Please check the path.")

        self.image_paths = [os.path.join(abs_path, file) for file in os.listdir(abs_path)
                            if file.lower().endswith(('.png', '.jpg', '.jpeg'))]  

        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {abs_path}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert('RGB')

        image = self.transform(image)

        return image

def load_images(train_folder_path):
    dataset = ImageFolderDataset(train_folder_path)

    image_tensor = torch.stack([dataset[i] for i in range(len(dataset))])

    return image_tensor

# Updated to use relative paths
train_folder_path = './braindataset/Training'
test_folder_path = './braindataset/Testing'

glioma_train_tensor = load_images(os.path.join(train_folder_path, "glioma"))
meningioma_train_tensor = load_images(os.path.join(train_folder_path, "meningioma"))
notumor_train_tensor = load_images(os.path.join(train_folder_path, "notumor"))
pituitary_train_tensor = load_images(os.path.join(train_folder_path, "pituitary"))

glioma_test_tensor = load_images(os.path.join(test_folder_path, "glioma"))
meningioma_test_tensor = load_images(os.path.join(test_folder_path, "meningioma"))
notumor_test_tensor = load_images(os.path.join(test_folder_path, "notumor"))
pituitary_test_tensor = load_images(os.path.join(test_folder_path, "pituitary"))