import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders
from model import FakeFaceCNN

# Paths
data_dir = r"data\real_vs_fake\real_vs_fake"  # adjust if needed

# Load data
train_loader, val_loader, test_loader = get_dataloaders(data_dir)


# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FakeFaceCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # or whatever you choose

for epoch in range(num_epochs):
    print(f"\nStarting Epoch {epoch+1}/{num_epochs}...")
    
    # ---- Training Phase ----
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    print(f"Train Loss: {epoch_loss:.4f}  Train Acc: {epoch_acc:.4f}")

    # ---- Validation Phase ----
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

    val_loss /= len(val_loader.dataset)
    val_acc = val_corrects.double() / len(val_loader.dataset)
    print(f"Val Loss:   {val_loss:.4f}  Val Acc: {val_acc:.4f}")
