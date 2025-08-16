from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import random
import os

def get_balanced_subset(dataset, per_class_count):
    """Return indices for a balanced subset from ImageFolder dataset."""
    targets = dataset.targets  # class indices
    indices_per_class = {cls: [] for cls in set(targets)}

    # Collect indices per class
    for idx, label in enumerate(targets):
        indices_per_class[label].append(idx)

    # Randomly sample per_class_count from each class
    final_indices = []
    for cls, idxs in indices_per_class.items():
        sampled = random.sample(idxs, per_class_count)
        final_indices.extend(sampled)

    random.shuffle(final_indices)
    return final_indices

def get_dataloaders(data_dir, train_per_class=10000, val_per_class=2000, test_per_class=2000, batch_size=32):
    train_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=val_tf)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_tf)

    # Create balanced subsets
    train_indices = get_balanced_subset(train_ds, train_per_class)
    val_indices = get_balanced_subset(val_ds, val_per_class)
    test_indices = get_balanced_subset(test_ds, test_per_class)

    train_ds = Subset(train_ds, train_indices)
    val_ds = Subset(val_ds, val_indices)
    test_ds = Subset(test_ds, test_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader
