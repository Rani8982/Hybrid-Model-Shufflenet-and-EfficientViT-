.from torchinfo import summary

# Define the model with the number of classes
num_classes = 6  # Change this based on your dataset
model = CombinedModel(num_classes=num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Print model summary
summary(model, input_size=(1,3, 224, 224))  # Batch size 1, 3 color channels, 224x224 resolution
