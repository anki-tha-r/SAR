import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from unet import UNet

# Paths
IMAGE_DIR = 'dataset/images'
MASK_DIR = 'dataset/masks'
CHECKPOINT_DIR = 'checkpoints'

# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Custom Dataset
class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # Only keep images for which a corresponding mask file exists
        self.images = [img for img in os.listdir(image_dir) if os.path.isfile(os.path.join(mask_dir, img))]
        print(f"✅ Found {len(self.images)} image-mask pairs.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"❌ Image not found: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"❌ Mask not found: {mask_path}")

        image = cv2.resize(image, (256, 256)) / 255.0
        mask = cv2.resize(mask, (256, 256)) / 255.0

        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)

        return image, mask

# Training Function
def train():
    print("🚀 Starting training...")
    dataset = RoadDataset(IMAGE_DIR, MASK_DIR)
    
    if len(dataset) == 0:
        print("⚠️ No data found! Check dataset paths or file names.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = UNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    start_epoch = 0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'unet_checkpoint.pth')

    # Resume training if checkpoint exists
    if os.path.exists(checkpoint_path):
        print("📦 Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"🔁 Resuming training from epoch {start_epoch}...")

    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        print(f"\n📘 Epoch {epoch + 1}/{EPOCHS}")

        for batch_idx, (images, masks) in enumerate(loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(loader):
                print(f"  🔁 Batch {batch_idx+1}/{len(loader)} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"✅ Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.4f}")

        # Save current checkpoint
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(CHECKPOINT_DIR, 'unet_checkpoint.pth'))

        # Save epoch-specific checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(CHECKPOINT_DIR, f'unet_checkpoint_epoch_{epoch+1}.pth'))

    # Save final model
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'unet_model.pth'))
    print("\n🏁 Training completed and final model saved.")

if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        print(f"❌ Error occurred during training: {e}")
