import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from torchvision.datasets import ImageFolder

# Load dataset (without transforms, so we can apply them manually)
dataset = ImageFolder(root='/kaggle/input/driver2x/nic')

# Get one sample image (first image in the dataset)
sample_image_path, _ = dataset.samples[0]  # Path to first image
sample_image = Image.open(sample_image_path).convert("RGB")  # Ensure RGB mode

# Define individual transformations
transformations = {
    "Original": lambda img: img,
    "Resize (224x224)": transforms.Resize((224, 224)),
    "Horizontal Flip": lambda img: ImageOps.mirror(img),  # Ensure correct horizontal flip
    "Rotation (40°)": transforms.RandomRotation(40),
    "Color Jitter": transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    "Affine Transform": transforms.RandomAffine(degrees=40, scale=(0.3, 1.1), shear=0.15),
    "Gaussian Blur": transforms.GaussianBlur(kernel_size=5),
    "Grayscale": transforms.Grayscale(num_output_channels=3),
}

# Function to apply mirror trick (concatenates original and flipped)
def mirror_trick(image):
    flipped = ImageOps.mirror(image)  # Horizontally flip the image
    mirrored = Image.new("RGB", (image.width * 2, image.height))  # Create a new blank canvas (twice the width)
    mirrored.paste(image, (0, 0))  # Paste original on the left
    mirrored.paste(flipped, (image.width, 0))  # Paste flipped on the right
    return mirrored

# Apply each transformation
transformed_images = {}
for name, transform in transformations.items():
    transformed_images[name] = transform(sample_image) if callable(transform) else transform(sample_image)

# Apply mirror trick separately
transformed_images["Mirror Trick"] = mirror_trick(sample_image)

# Display images
fig, axes = plt.subplots(3, 3, figsize=(15, 12))  # 3x3 grid
for ax, (name, img) in zip(axes.flat, transformed_images.items()):
    ax.imshow(img if isinstance(img, Image.Image) else np.array(img))  # Convert PIL images properly
    ax.set_title(name)
    ax.axis('off')

plt.tight_layout()
plt.show()
