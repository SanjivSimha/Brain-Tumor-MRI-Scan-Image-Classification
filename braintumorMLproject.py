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


class ImageFolderDataset(Dataset):
    def __init__(self, image_paths):
        # here, you should set any important vairables you want to use in __len__ or __getitem__
        self.image_paths = image_paths

        self.transform = transforms.Compose([
          transforms.ToTensor(),
          # put other transforms here! https://pytorch.org/vision/stable/transforms.html#v2-api-ref
          # some useful ones might be resizing or cropping, consider also doing random augmentations
          # to make your model more robust.
        ])


    # this should just return the length of the dataset, without processing or opening any images
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # we get the image path from the index
        image_path = self.image_paths[idx]
        print(image_path)
        # use the PIL class to get the image
        image = Image.open(image_path).convert('RGB')

        # run transformations on image
        image = self.transform(image)
    
        return image
    

images_training_glioma = []

train_giloma = ImageFolderDataset('WINTER2025AIPROJECT/braindataset/Testing/glioma')
train_meningioma = ImageFolderDataset('WINTER2025AIPROJECT/braindataset/Testing/train_meningioma')
train_notumor = ImageFolderDataset('WINTER2025AIPROJECT/braindataset/Testing/notumor')
train_pituitary = ImageFolderDataset('WINTER2025AIPROJECT/braindataset/Testing/pituitary')

len_giloma = len(train_giloma)
len_meningioma = len(train_meningioma)
len_notumor = len(train_notumor)
len_pituitary = len(train_pituitary)

print(f"Number of glioma images: {len_giloma}")
print(f"Number of meningioma images: {len_meningioma}")
print(f"Number of no tumor images: {len_notumor}")
print(f"Number of pituitary images: {len_pituitary}")

print(images_training_glioma)