import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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

train_images = torch.cat([glioma_train_tensor, meningioma_train_tensor, notumor_train_tensor, pituitary_train_tensor], dim=0)
train_labels = torch.cat([
    torch.full((glioma_train_tensor.size(0),), 0),  # Label 0 for glioma
    torch.full((meningioma_train_tensor.size(0),), 1),  # Label 1 for meningioma
    torch.full((notumor_train_tensor.size(0),), 2),  # Label 2 for notumor
    torch.full((pituitary_train_tensor.size(0),), 3)  # Label 3 for pituitary
], dim=0)

test_images = torch.cat([glioma_test_tensor, meningioma_test_tensor, notumor_test_tensor, pituitary_test_tensor], dim=0)
test_labels = torch.cat([
    torch.full((glioma_test_tensor.size(0),), 0),  # Label 0 for glioma
    torch.full((meningioma_test_tensor.size(0),), 1),  # Label 1 for meningioma
    torch.full((notumor_test_tensor.size(0),), 2),  # Label 2 for notumor
    torch.full((pituitary_test_tensor.size(0),), 3)  # Label 3 for pituitary
], dim=0)

train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), 
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), 
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), 
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 4)
        )

    def forward(self, x):
        return self.model(x)

# creating the nerual network, optimizer, and loss function
classifier = Model().to('cpu')
optimizer = Adam(classifier.parameters(), lr = 1e-3)
loss_function = nn.CrossEntropyLoss()

# training loop
# epochs = 10

# for epoch in range(epochs):
#     classifier.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     for images, labels in train_loader:
#         images, labels = images.to('cpu'), labels.to('cpu')

#         optimizer.zero_grad()

#         outputs = classifier(images)
#         loss = loss_function(outputs, labels)

#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     # display epochs
#     print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.2f}, Accuracy: {100 * correct / total:.2f}%")

# # save model in .pth after training
# torch.save(classifier.state_dict(), 'model_after_10_epochs.pth')
# print("model saved")

# load up saved model to use in testing loop
model = Model().to('cpu')
model.load_state_dict(torch.load('model_after_10_epochs.pth'))
model.eval()

# testing loop

with torch.no_grad():
    total_predictions = []
    total_labels = []

    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1) 
        total_predictions.extend(predictions.cpu().numpy())  
        total_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(total_labels, total_predictions)
    print(f'Test Accuracy: {accuracy * 100:.3f}%')

def plot_confusion_matrix(y_true, y_pred, class_names):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_class_accuracies(y_true, y_pred, class_names):
    # Convert to numpy arrays if they aren't already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    class_accuracies = []
    for i in range(len(class_names)):
        # Find all indices where true label is class i
        class_indices = np.where(y_true == i)[0]
        if len(class_indices) > 0:  # Only calculate if we have samples
            class_acc = accuracy_score(y_true[class_indices], y_pred[class_indices])
            class_accuracies.append(class_acc * 100)
        else:
            class_accuracies.append(0)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_accuracies)
    plt.title('Class-wise Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Class')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('class_accuracies.png')
    plt.close()

def plot_prediction_distribution(y_true, y_pred, class_names):
    plt.figure(figsize=(12, 5))
    
    # Convert to numpy arrays if they aren't already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Plot true distribution
    plt.subplot(1, 2, 1)
    true_dist = [np.sum(y_true == i) for i in range(len(class_names))]
    plt.bar(class_names, true_dist)
    plt.title('True Class Distribution')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    
    # Plot predicted distribution
    plt.subplot(1, 2, 2)
    pred_dist = [np.sum(y_pred == i) for i in range(len(class_names))]
    plt.bar(class_names, pred_dist)
    plt.title('Predicted Class Distribution')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()

class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Convert predictions and labels to numpy arrays
total_predictions = np.array(total_predictions)
total_labels = np.array(total_labels)

# Create visualizations
plot_confusion_matrix(total_labels, total_predictions, class_names)
plot_class_accuracies(total_labels, total_predictions, class_names)
plot_prediction_distribution(total_labels, total_predictions, class_names)

# Print detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(total_labels, total_predictions, target_names=class_names))