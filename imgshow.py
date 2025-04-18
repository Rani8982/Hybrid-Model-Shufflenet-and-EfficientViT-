import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np

def imshow(img):
    img = img.numpy().transpose((1, 2, 0))  # Convert from tensor to numpy and rearrange dimensions
    img = np.clip(img, 0, 1)  # Ensure values are in valid range
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Define transformations (resize, normalize, and convert to tensor)
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # Resize to match LightCNN input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset (replace 'dataset_root' with actual path)
dataset_root = "/kaggle/input/driver2x/nic"
dataset = ImageFolder(root=dataset_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Get a batch of images
images, labels = next(iter(dataloader))

# Display 10 images
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    img = images[i].numpy().transpose((1, 2, 0))
    img = (img * 0.5) + 0.5  # Undo normalization
    ax.imshow(img)
    ax.axis("off")
plt.show()
