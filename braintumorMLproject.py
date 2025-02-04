from PIL import Image
import os
import numpy as np

folder_path = 'braindataset/Testing/glioma'

images_training_glioma = []

for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        file_path = os.path.join(folder_path, filename)
        img = Image.open(file_path)
        images_training_glioma.append(img)

print(images_training_glioma)