class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """ Stores the activation maps (feature maps) from the forward pass. """
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        """ Stores the gradients from the backward pass. """
        self.gradients = grad_output[0]  # Gradient of the activation

    def generate_heatmap(self, input_tensor, target_class=None):
        input_tensor = input_tensor.unsqueeze(0).to(device)
        output = self.model(input_tensor)

        # If no target class is provided, pick the predicted class
        if target_class is None:
            target_class = output.argmax().item()

        # Compute gradients for the target class
        self.model.zero_grad()
        output[:, target_class].backward()

        # Get activations and gradients
        activations = self.activations
        gradients = self.gradients

        # Compute global average pooling on gradients
        alpha = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activations
        cam = (alpha * activations).sum(dim=1).squeeze()
        cam = torch.relu(cam).detach().cpu().numpy()

        # Normalize and resize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
        cam = cv2.resize(cam, (224, 224))

# Convert heatmap to color
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        return cam
