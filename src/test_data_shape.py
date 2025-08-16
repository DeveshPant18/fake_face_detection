# test_data_shape.py
from data import get_dataloaders
from model import FakeFaceCNN
import torch

data_dir = "data/real_vs_fake/real_vs_fake"
train_loader, val_loader, test_loader = get_dataloaders(data_dir)

model = FakeFaceCNN()

# Get one batch
images, labels = next(iter(train_loader))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

# Forward pass
outputs = model(images)
print("Output shape:", outputs.shape)
