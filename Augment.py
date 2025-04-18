import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder

# Load dataset (without transforms, so we can apply them manually)
dataset = ImageFolder(root='/kaggle/input/driver2x/nic')

# Get one sample image (first image in the dataset)
sample_image_path, _ = dataset.samples[0]  # Path to first image
sample_image = Image.open(sample_image_path).convert("RGB")  # Open image & ensure RGB mode

# Define individual transformations
transformations = {
    "Original": transforms.Compose([]),
    "Resize (224x224)": transforms.Resize((224, 224)),
    "Horizontal Flip": transforms.RandomHorizontalFlip(p=1),  # Always flip
    "Rotation (40°)": transforms.RandomRotation(40),
    "Color Jitter": transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    "Affine Transform": transforms.RandomAffine(degrees=40, scale=(0.3, 1.1), shear=0.15),
    "Gaussian Blur": transforms.GaussianBlur(kernel_size=5),
    "Grayscale": transforms.Grayscale(num_output_channels=3),
    "Normalization": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
}

# Apply each transformation
transformed_images = {}
for name, transform in transformations.items():
    if name == "Original":
        transformed_images[name] = sample_image
    elif name == "Normalization":
        img = transform(sample_image).numpy().transpose((1, 2, 0))  # Convert tensor to numpy
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean  # De-normalize
        img = np.clip(img, 0, 1)  # Clip values
        transformed_images[name] = img
    else:
        transformed_images[name] = transform(sample_image)

# Display images
fig, axes = plt.subplots(3, 3, figsize=(12, 10))  # 3x3 grid
for ax, (name, img) in zip(axes.flat, transformed_images.items()):
    if isinstance(img, torch.Tensor):  # If tensor, convert to numpy
        img = img.numpy().transpose((1, 2, 0))
    ax.imshow(img)
    ax.set_title(name)
    ax.axis('off')

plt.tight_layout()
plt.show()
