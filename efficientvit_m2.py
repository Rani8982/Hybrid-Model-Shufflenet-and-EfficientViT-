import timm

net = timm.create_model('efficientvit_m2', pretrained=False)
print(net)  # Prints the model architecture
