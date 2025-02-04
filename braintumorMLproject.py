import os
import numpy as np
# PIL, or Python Image Library, helps us with processing the image file
from PIL import Image

# torch is used for tensors and machine learning
import torch
# torchvision.transforms, or torchvision.trasnforms.v2, allow us to transform each image, for example
# resize, crop, rotate, convert to tensor, blur, etc.
from torchvision.transforms import v2 as transforms

# torch Dataset and Dataloader allow us to effiecently process our dataset.
from torch.utils.data import Dataset, DataLoader


import os
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import v2 as transforms
from torch.utils.data import Dataset, DataLoader

class ImageFolderDataset():
    def __init__(self, folder_path):
        # Get all image file paths
        self.image_paths = [os.path.join(folder_path, file) for file in os.listdir(os.path.abspath(folder_path))]

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
    

images_training_glioma = []

train_giloma = ImageFolderDataset('../braindataset/Testing/glioma')
train_meningioma = ImageFolderDataset('../braindataset/Testing/meningioma')
train_notumor = ImageFolderDataset('../WINTER2025AIPROJECT/braindataset/Testing/notumor')
train_pituitary = ImageFolderDataset('../WINTER2025AIPROJECT/braindataset/Testing/pituitary')


len_giloma = len(train_giloma)
len_meningioma = len(train_meningioma)
len_notumor = len(train_notumor)
len_pituitary = len(train_pituitary)

print(f"Number of glioma images: {len_giloma}")
print(f"Number of meningioma images: {len_meningioma}")
print(f"Number of no tumor images: {len_notumor}")
print(f"Number of pituitary images: {len_pituitary}")


print(images_training_glioma)