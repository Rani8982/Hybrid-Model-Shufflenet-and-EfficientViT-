device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
num_classes = 7  # Adjust based on your dataset
model = CombinedModel(num_classes).to(device)
model.eval()

# Select the correct convolutional layer for Grad-CAM
target_layer = model.feature_extractor_shufflenet.stage4[-1]  # Last conv layer of ShuffleNet

# Initialize Grad-CAM
gradcam = GradCAM(model, target_layer)
