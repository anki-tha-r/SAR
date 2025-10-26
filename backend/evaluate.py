import torch
from unet import UNet
from utils import RoadDataset  # Ensure this exists and is correctly implemented
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import os

# Load the trained model
model = UNet()
model.load_state_dict(torch.load('checkpoints/unet_model.pth'))
model.eval()  # Set the model to evaluation mode

# Define dataset and DataLoader
val_image_dir = os.path.join("dataset", "val", "images")
val_mask_dir = os.path.join("dataset", "val", "masks")

dataset = RoadDataset(val_image_dir, val_mask_dir)  # Make sure these folders have images/masks
data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

# Check if validation dataset is empty
if len(data_loader) == 0:
    print("❌ No validation data found. Please check 'dataset/val/images' and 'dataset/val/masks'.")
    exit()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluation loop
total_loss = 0.0
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss

# Run inference on the dataset
with torch.no_grad():
    for img, mask in data_loader:
        img, mask = img.to(device), mask.to(device)

        # Get the model's predictions
        pred = model(img)

        # Calculate loss
        loss = criterion(pred, mask)
        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)

print(f"✅ Evaluation complete! Average BCE Loss: {avg_loss:.4f}")
