image_path = "/kaggle/input/driver2x/nic/SU/03_SU_s02_058.jpg"
image = Image.open(image_path).convert("RGB")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Preprocess image
input_tensor = transform(image).to(device)
# Generate Grad-CAM heatmap
heatmap = gradcam.generate_heatmap(input_tensor)

# Convert heatmap to color
heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

# Convert original image to NumPy
original = np.array(image.resize((224, 224)))

# Overlay heatmap on original image
overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

# Display results
plt.figure(figsize=(10, 5))

# Show original image
plt.subplot(1, 2, 1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

# Show Grad-CAM overlay
plt.subplot(1, 2, 2)
plt.imshow(overlay)
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.show()
