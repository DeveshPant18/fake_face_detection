import torch
from data import get_dataloaders

if __name__ == "__main__":
    data_dir = "data/real_vs_fake/real_vs_fake"
    train_loader, val_loader, test_loader = get_dataloaders(data_dir)

    # Check one batch
    images, labels = next(iter(train_loader))
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")
    print(f"Unique labels in batch: {labels.unique()}")  # should be 0 and 1
