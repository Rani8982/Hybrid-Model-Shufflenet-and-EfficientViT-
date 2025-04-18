from torchsummary import summary
import torch
import torchvision.models as models

# Load ShuffleNet v2 (1.0x) with pre-trained weights
model = models.shufflenet_v2_x1_0(weights='ShuffleNet_V2_X1_0_Weights.DEFAULT')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Print model summary
summary(model, (3, 224, 224))  # Input size: (channels, height, width)
