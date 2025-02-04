import os
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import v2 as transforms
from torch.utils.data import Dataset, DataLoader

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path):
        abs_path = os.path.abspath(folder_path)

        # Check if the directory exists
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Error: The folder path '{abs_path}' does not exist. Please check the path.")

        # Get all image file paths
        self.image_paths = [os.path.join(abs_path, file) for file in os.listdir(abs_path)
                            if file.lower().endswith(('.png', '.jpg', '.jpeg'))]  # Ensure only images are selected

        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {abs_path}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def getitem(self, idx):
        image_path = self.image_paths[idx]

        # Open image
        image = Image.open(image_path).convert('RGB')

        # Apply transformations
        image = self.transform(image)

        return image
    
def load_train_images(train_folder_path):
    dataset = ImageFolderDataset(train_folder_path)
    
    image_tensor = torch.tensor([dataset.getitem(i) for i in range(len(dataset))])

train_folder_path = './braindataset/Training'

giloma_train_tensot = load_train_images(train_folder_path + "/giloma")
meningioma_train = load_train_images(train_folder_path + "/meningioma")
notumor_train = load_train_images(train_folder_path + "/notumor")
pituitary_train = load_train_images(train_folder_path + "/pituitary")




