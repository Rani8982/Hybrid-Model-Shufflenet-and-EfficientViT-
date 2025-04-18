# Load model safely
checkpoint = torch.load('/kaggle/working/Test_model.t7', map_location=device)

# Handle state_dict mismatch
model_dict = net.state_dict()
pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in model_dict}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict, strict=False)

# Move model to device
net.to(device)
