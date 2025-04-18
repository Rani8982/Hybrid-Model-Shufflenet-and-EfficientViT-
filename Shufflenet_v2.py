import torch
import torchvision.models as models

model = models.shufflenet_v2_x1_0(weights='ShuffleNet_V2_X1_0_Weights.DEFAULT')
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last FC layer
