# # # import os
# # # import cv2
# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim
# # # from torch.utils.data import DataLoader, Dataset
# # # from unet import UNet

# # # # Paths
# # # IMAGE_DIR = 'dataset/images'
# # # MASK_DIR = 'dataset/masks'
# # # CHECKPOINT_DIR = 'land_checkpoints'

# # # # Hyperparameters
# # # EPOCHS = 20
# # # BATCH_SIZE = 4
# # # LEARNING_RATE = 1e-4
# # # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# # # # Custom Dataset
# # # class LandDataset(Dataset):
# # #     def __init__(self, image_dir, mask_dir):
# # #         self.image_dir = image_dir
# # #         self.mask_dir = mask_dir

# # #         # Only keep images for which a corresponding mask file exists
# # #         self.images = [img for img in os.listdir(image_dir) if os.path.isfile(os.path.join(mask_dir, img))]
# # #         print(f"✅ Found {len(self.images)} image-mask pairs.")

# # #     def __len__(self):
# # #         return len(self.images)

# # #     def __getitem__(self, idx):
# # #         img_name = self.images[idx]
# # #         img_path = os.path.join(self.image_dir, img_name)
# # #         mask_path = os.path.join(self.mask_dir, img_name)

# # #         image = cv2.imread(img_path)
# # #         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# # #         if image is None:
# # #             raise FileNotFoundError(f"❌ Image not found: {img_path}")
# # #         if mask is None:
# # #             raise FileNotFoundError(f"❌ Mask not found: {mask_path}")

# # #         image = cv2.resize(image, (256, 256)) / 255.0
# # #         mask = cv2.resize(mask, (256, 256)) / 255.0

# # #         image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
# # #         mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)

# # #         return image, mask

# # # # Training Function
# # # def train():
# # #     print("🚀 Starting training...")
# # #     dataset = LandDataset(IMAGE_DIR, MASK_DIR)
    
# # #     if len(dataset) == 0:
# # #         print("⚠️ No data found! Check dataset paths or file names.")
# # #         return

# # #     loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# # #     model = UNet().to(DEVICE)
# # #     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# # #     criterion = nn.BCELoss()

# # #     start_epoch = 0
# # #     checkpoint_path = os.path.join(CHECKPOINT_DIR, 'unet_checkpoint.pth')

# # #     # Resume training if checkpoint exists
# # #     if os.path.exists(checkpoint_path):
# # #         print("📦 Loading checkpoint...")
# # #         checkpoint = torch.load(checkpoint_path)
# # #         model.load_state_dict(checkpoint['model_state_dict'])
# # #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# # #         start_epoch = checkpoint['epoch'] + 1
# # #         print(f"🔁 Resuming training from epoch {start_epoch}...")

# # #     # Training loop
# # #     for epoch in range(start_epoch, EPOCHS):
# # #         model.train()
# # #         total_loss = 0
# # #         print(f"\n📘 Epoch {epoch + 1}/{EPOCHS}")

# # #         for batch_idx, (images, masks) in enumerate(loader):
# # #             images, masks = images.to(DEVICE), masks.to(DEVICE)

# # #             preds = model(images)
# # #             loss = criterion(preds, masks)

# # #             optimizer.zero_grad()
# # #             loss.backward()
# # #             optimizer.step()

# # #             total_loss += loss.item()
# # #             if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(loader):
# # #                 print(f"  🔁 Batch {batch_idx+1}/{len(loader)} - Loss: {loss.item():.4f}")

# # #         avg_loss = total_loss / len(loader)
# # #         print(f"✅ Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.4f}")

# # #         # Save current checkpoint
# # #         os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# # #         torch.save({
# # #             'epoch': epoch,
# # #             'model_state_dict': model.state_dict(),
# # #             'optimizer_state_dict': optimizer.state_dict()
# # #         }, os.path.join(CHECKPOINT_DIR, 'land_unet_checkpoint.pth'))

# # #         # Save epoch-specific checkpoint
# # #         torch.save({
# # #             'epoch': epoch,
# # #             'model_state_dict': model.state_dict(),
# # #             'optimizer_state_dict': optimizer.state_dict()
# # #         }, os.path.join(CHECKPOINT_DIR, f'land_unet_checkpoint_epoch_{epoch+1}.pth'))

# # #     # Save final model
# # #     torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'land_unet_model.pth'))
# # #     print("\n🏁 Training completed and final model saved.")

# # # if __name__ == '__main__':
# # #     try:
# # #         train()
# # #     except Exception as e:
# # #         print(f"❌ Error occurred during training: {e}")


# """
# Train script for DeepLabV3+ (smp). Example:
# python train.py --data_dir dataset --epochs 20 --batch_size 4 --save_dir checkpoints
# """

# # import os, glob

# # print("Current working directory:", os.getcwd())
# # print("DATA_DIR:", dataset)
# # print("Images path:", os.path.join(DATA_DIR, "images"))
# # print("Masks path:", os.path.join(DATA_DIR, "masks"))

# # images_list = sorted(glob.glob(os.path.join(DATA_DIR, "images", "*.*")))
# # masks_list = sorted(glob.glob(os.path.join(DATA_DIR, "masks", "*.*")))

# # print(f"Found {len(images_list)} images and {len(masks_list)} masks")
# # if len(images_list) > 0:
# #     print("First 3 images:", images_list[:3])
# # if len(masks_list) > 0:
# #     print("First 3 masks:", masks_list[:3])


# import os
# import random
# import argparse
# from tqdm import tqdm

# import torch
# from torch.utils.data import DataLoader
# import albumentations as album
# import segmentation_models_pytorch as smp
# from segmentation_models_pytorch.losses import DiceLoss
# from segmentation_models_pytorch.utils.metrics import iou_score



# from utils import list_images_and_masks, LandCoverDataset, CLASS_RGB


# def get_training_augmentation():
#     return album.Compose([
#         album.RandomCrop(height=512, width=512),
#         album.HorizontalFlip(p=0.5),
#         album.VerticalFlip(p=0.5),
#     ])


# def get_validation_augmentation():
#     return album.Compose([
#         album.CenterCrop(height=512, width=512),
#     ])


# def to_tensor(x, **kwargs):
#     return x.transpose(2, 0, 1).astype('float32')


# def get_preprocessing(preprocessing_fn=None):
#     transforms = []
#     if preprocessing_fn:
#         transforms.append(album.Lambda(image=preprocessing_fn))
#     transforms.append(album.Lambda(image=to_tensor, mask=to_tensor))
#     return album.Compose(transforms)


# def train(args):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     images_list, masks_list = list_images_and_masks(os.path.join(args.data_dir, 'images'),
#                                                     os.path.join(args.data_dir, 'masks'))
#     assert len(images_list) > 0, "No images found. Place images in dataset/images and masks in dataset/masks"

#     # 90/10 split
#     indices = list(range(len(images_list)))
#     random.seed(42)
#     random.shuffle(indices)
#     split = int(0.9 * len(indices))
#     train_idx, val_idx = indices[:split], indices[split:]

#     train_images = [images_list[i] for i in train_idx]
#     train_masks = [masks_list[i] for i in train_idx]
#     val_images = [images_list[i] for i in val_idx]
#     val_masks = [masks_list[i] for i in val_idx]

#     ENCODER = args.encoder
#     ENCODER_WEIGHTS = 'imagenet'
#     CLASSES = list(range(len(CLASS_RGB)))
#     ACTIVATION = 'sigmoid' if len(CLASSES) > 1 else None

#     model = smp.DeepLabV3Plus(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=len(CLASSES), activation=ACTIVATION)
#     model.to(device)

#     preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

#     train_dataset = LandCoverDataset(train_images, train_masks, class_rgb_values=CLASS_RGB,
#                                      augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
#     val_dataset = LandCoverDataset(val_images, val_masks, class_rgb_values=CLASS_RGB,
#                                    augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))

#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
#     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

#     loss_fn = DiceLoss(mode='binary')  # or mode='multiclass' if you have multiple classes
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#     os.makedirs(args.save_dir, exist_ok=True)
#     best_iou = 0.0

#     for epoch in range(args.epochs):
#         model.train()
#         train_loss = 0.0
#         loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}] Train")
#         for imgs, masks in loop:
#             imgs = imgs.to(device)
#             masks = masks.to(device)
#             optimizer.zero_grad()
#             outputs = model(imgs)
#             loss_value = loss_fn(outputs, masks)
#             loss_value.backward()
#             optimizer.step()
#             train_loss += loss_value.item()
#             loop.set_postfix({'train_loss': train_loss / (loop.n + 1)})

# def compute_iou(preds, masks, threshold=0.5):
#     """
#     preds: torch tensor with logits or probabilities
#     masks: ground truth mask tensor
#     """
#     preds = torch.sigmoid(preds)  # if logits
#     preds = (preds > threshold).float()
    
#     intersection = (preds * masks).sum(dim=(1,2,3))
#     union = (preds + masks).sum(dim=(1,2,3)) - intersection
#     iou = (intersection + 1e-6) / (union + 1e-6)
#     return iou.mean()


#         # # validation
#         # model.eval()
#         # val_loss = 0.0
#         # total_iou = 0.0
#         # cnt = 0
#         # with torch.no_grad():
#         #     for imgs, masks in tqdm(val_loader, desc='Validation'):
#         #         imgs = imgs.to(device)
#         #         masks = masks.to(device)
#         #         outputs = model(imgs)
#         #         val_loss += loss_fn(outputs, masks).item()
#         #         # compute IoU using smp metric (works with logits)
#         #         iou = iou_score(outputs, masks)
#         #         total_iou += iou.item()

#         #         cnt += 1

#         # avg_val_iou = total_iou / max(1, cnt)
#         # avg_val_loss = val_loss / max(1, cnt)
#         # print(f"Epoch {epoch}: val_loss={avg_val_loss:.4f}, val_iou={avg_val_iou:.4f}")

#         # validation
#         model.eval()
#         val_loss = 0.0
#         total_iou = 0.0
#         cnt = 0
#         with torch.no_grad():
#             for imgs, masks in tqdm(val_loader, desc='Validation'):
#                 imgs = imgs.to(device)
#                 masks = masks.to(device)

#                 outputs = model(imgs)
#                 val_loss += loss_fn(outputs, masks).item()

#                 # compute IoU (use functional iou_score)
#                 batch_iou = iou_score(outputs, masks)
#                 total_iou += batch_iou.item()
#                 cnt += 1

#         avg_val_iou = total_iou / max(1, cnt)
#         avg_val_loss = val_loss / max(1, cnt)
#         print(f"Epoch {epoch}: val_loss={avg_val_loss:.4f}, val_iou={avg_val_iou:.4f}")


#         # # Initialize IoU metric
#         # iou_metric = IoU(threshold=0.5)

#         # # validation
#         # model.eval()
#         # val_loss = 0.0
#         # total_iou = 0.0
#         # cnt = 0
#         # with torch.no_grad():
#         #     for imgs, masks in tqdm(val_loader, desc='Validation'):
#         #         imgs = imgs.to(device)
#         #         masks = masks.to(device)

#         #         outputs = model(imgs)
#         #         val_loss += loss_fn(outputs, masks).item()

#         #         # compute IoU
#         #         batch_iou = iou_metric(outputs, masks)
#         #         total_iou += batch_iou.item()
#         #         cnt += 1

#         # avg_val_iou = total_iou / max(1, cnt)
#         # avg_val_loss = val_loss / max(1, cnt)
#         # print(f"Epoch {epoch}: val_loss={avg_val_loss:.4f}, val_iou={avg_val_iou:.4f}")


#         # save checkpoint
#         ckpt_path = os.path.join(args.save_dir, f"epochs+{epoch}.pth")
#         torch.save(model.state_dict(), ckpt_path)
#         print(f"Saved checkpoint: {ckpt_path}")

#         if avg_val_iou > best_iou:
#             best_iou = avg_val_iou
#             best_path = os.path.join(args.save_dir, "best_model.pth")
#             torch.save(model.state_dict(), best_path)
#             print(f"Saved best model: {best_path}")

#     print("Training finished.")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_dir', default='dataset', help='dataset folder (images/ masks/)')
#     parser.add_argument('--save_dir', default='checkpoints', help='where to save checkpoints')
#     parser.add_argument('--epochs', type=int, default=20)
#     parser.add_argument('--batch_size', type=int, default=4)
#     parser.add_argument('--lr', type=float, default=8e-5)
#     parser.add_argument('--encoder', default='resnet50')
#     parser.add_argument('--num_workers', type=int, default=4)
#     args = parser.parse_args()
#     train(args)

# # import os
# # import random
# # import argparse
# # from tqdm import tqdm

# # import torch
# # from torch.utils.data import DataLoader
# # import albumentations as album
# # import segmentation_models_pytorch as smp
# # from segmentation_models_pytorch.losses import DiceLoss
# # from segmentation_models_pytorch.utils.metrics import iou_score

# # from utils import list_images_and_masks, LandCoverDataset, CLASS_RGB


# # # --------------------- Augmentations --------------------- #
# # def get_training_augmentation():
# #     return album.Compose([
# #         album.RandomCrop(height=512, width=512),
# #         album.HorizontalFlip(p=0.5),
# #         album.VerticalFlip(p=0.5),
# #     ])


# # def get_validation_augmentation():
# #     return album.Compose([
# #         album.CenterCrop(height=512, width=512),
# #     ])


# # # --------------------- Preprocessing --------------------- #
# # def to_tensor(x, **kwargs):
# #     return x.transpose(2, 0, 1).astype('float32')


# # def get_preprocessing(preprocessing_fn=None):
# #     transforms = []
# #     if preprocessing_fn:
# #         transforms.append(album.Lambda(image=preprocessing_fn))
# #     transforms.append(album.Lambda(image=to_tensor, mask=to_tensor))
# #     return album.Compose(transforms)


# # # --------------------- Training Function --------------------- #
# # def train(args):
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# #     # Load images and masks
# #     images_list, masks_list = list_images_and_masks(
# #         os.path.join(args.data_dir, 'images'),
# #         os.path.join(args.data_dir, 'masks')
# #     )
# #     assert len(images_list) > 0, "No images found. Place images in dataset/images and dataset/masks"

# #     # 90/10 train/val split
# #     indices = list(range(len(images_list)))
# #     random.seed(42)
# #     random.shuffle(indices)
# #     split = int(0.9 * len(indices))
# #     train_idx, val_idx = indices[:split], indices[split:]

# #     train_images = [images_list[i] for i in train_idx]
# #     train_masks = [masks_list[i] for i in train_idx]
# #     val_images = [images_list[i] for i in val_idx]
# #     val_masks = [masks_list[i] for i in val_idx]

# #     # Model setup
# #     ENCODER = args.encoder
# #     ENCODER_WEIGHTS = 'imagenet'
# #     CLASSES = list(range(len(CLASS_RGB)))
# #     ACTIVATION = 'sigmoid' if len(CLASSES) > 1 else None

# #     model = smp.DeepLabV3Plus(
# #         encoder_name=ENCODER,
# #         encoder_weights=ENCODER_WEIGHTS,
# #         classes=len(CLASSES),
# #         activation=ACTIVATION
# #     )
# #     model.to(device)

# #     preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# #     # Datasets and loaders
# #     train_dataset = LandCoverDataset(
# #         train_images, train_masks, class_rgb_values=CLASS_RGB,
# #         augmentation=get_training_augmentation(),
# #         preprocessing=get_preprocessing(preprocessing_fn)
# #     )
# #     val_dataset = LandCoverDataset(
# #         val_images, val_masks, class_rgb_values=CLASS_RGB,
# #         augmentation=get_validation_augmentation(),
# #         preprocessing=get_preprocessing(preprocessing_fn)
# #     )

# #     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
# #     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

# #     # Loss and optimizer
# #     loss_fn = DiceLoss(mode='binary')  # use 'multiclass' if multiple classes
# #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# #     os.makedirs(args.save_dir, exist_ok=True)
# #     best_iou = 0.0

# #     # --------------------- Training Loop --------------------- #
# #     for epoch in range(args.epochs):
# #         model.train()
# #         train_loss = 0.0
# #         loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}] Train")
# #         for imgs, masks in loop:
# #             imgs = imgs.to(device)
# #             masks = masks.to(device)

# #             optimizer.zero_grad()
# #             outputs = model(imgs)
# #             loss_value = loss_fn(outputs, masks)
# #             loss_value.backward()
# #             optimizer.step()

# #             train_loss += loss_value.item()
# #             loop.set_postfix({'train_loss': train_loss / (loop.n + 1)})

# #         # --------------------- Validation --------------------- #
# #         model.eval()
# #         val_loss = 0.0
# #         total_iou = 0.0
# #         cnt = 0
# #         with torch.no_grad():
# #             for imgs, masks in tqdm(val_loader, desc='Validation'):
# #                 imgs = imgs.to(device)
# #                 masks = masks.to(device)

# #                 outputs = model(imgs)
# #                 val_loss += loss_fn(outputs, masks).item()

# #                 # IoU metric
# #                 batch_iou = iou_score(outputs, masks)
# #                 total_iou += batch_iou.item()
# #                 cnt += 1

# #         avg_val_iou = total_iou / max(1, cnt)
# #         avg_val_loss = val_loss / max(1, cnt)
# #         print(f"Epoch {epoch}: val_loss={avg_val_loss:.4f}, val_iou={avg_val_iou:.4f}")

# #         # Save checkpoints
# #         ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch}.pth")
# #         torch.save(model.state_dict(), ckpt_path)
# #         print(f"Saved checkpoint: {ckpt_path}")

# #         if avg_val_iou > best_iou:
# #             best_iou = avg_val_iou
# #             best_path = os.path.join(args.save_dir, "best_model.pth")
# #             torch.save(model.state_dict(), best_path)
# #             print(f"Saved best model: {best_path}")

# #     print("Training finished.")


# # # --------------------- Main --------------------- #
# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--data_dir', default='dataset', help='dataset folder (images/ masks/)')
# #     parser.add_argument('--save_dir', default='checkpoints', help='where to save checkpoints')
# #     parser.add_argument('--epochs', type=int, default=20)
# #     parser.add_argument('--batch_size', type=int, default=4)
# #     parser.add_argument('--lr', type=float, default=8e-5)
# #     parser.add_argument('--encoder', default='resnet50')
# #     parser.add_argument('--num_workers', type=int, default=4)
# #     args = parser.parse_args()
# #     train(args)


import os
import random
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import albumentations as album
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

from utils import list_images_and_masks, LandCoverDataset, CLASS_RGB


def get_training_augmentation():
    return album.Compose([
        album.RandomCrop(height=512, width=512),
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.5),
    ])


def get_validation_augmentation():
    return album.Compose([
        album.CenterCrop(height=512, width=512),
    ])


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    transforms = []
    if preprocessing_fn:
        transforms.append(album.Lambda(image=preprocessing_fn))
    transforms.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(transforms)


def compute_iou(preds, masks, threshold=0.5):
    """Compute IoU for binary segmentation"""
    preds = torch.sigmoid(preds)  # convert logits to probabilities
    preds = (preds > threshold).float()
    intersection = (preds * masks).sum(dim=(1,2,3))
    union = (preds + masks).sum(dim=(1,2,3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images_list, masks_list = list_images_and_masks(
        os.path.join(args.data_dir, 'images'),
        os.path.join(args.data_dir, 'masks')
    )
    assert len(images_list) > 0, "No images found. Place images in dataset/images and masks in dataset/masks"

    # 90/10 split
    indices = list(range(len(images_list)))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.9 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_images = [images_list[i] for i in train_idx]
    train_masks = [masks_list[i] for i in train_idx]
    val_images = [images_list[i] for i in val_idx]
    val_masks = [masks_list[i] for i in val_idx]

    ENCODER = args.encoder
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = list(range(len(CLASS_RGB)))
    ACTIVATION = 'sigmoid' if len(CLASSES) > 1 else None

    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION
    )
    model.to(device)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = LandCoverDataset(
        train_images, train_masks,
        class_rgb_values=CLASS_RGB,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn)
    )
    val_dataset = LandCoverDataset(
        val_images, val_masks,
        class_rgb_values=CLASS_RGB,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    loss_fn = DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_iou = 0.0

    start_epoch = 0
    if args.resume is not None and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint)
        start_epoch = int(os.path.basename(args.resume).split('+')[-1].split('.')[0]) + 1
        print(f"Starting from epoch {start_epoch}")


    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}] Train")
        for imgs, masks in loop:
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss_value = loss_fn(outputs, masks)
            loss_value.backward()
            optimizer.step()
            train_loss += loss_value.item()
            loop.set_postfix({'train_loss': train_loss / (loop.n + 1)})

        # validation
        model.eval()
        val_loss = 0.0
        total_iou = 0.0
        cnt = 0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc='Validation'):
                imgs = imgs.to(device)
                masks = masks.to(device)
                outputs = model(imgs)
                val_loss += loss_fn(outputs, masks).item()
                iou = compute_iou(outputs, masks)
                total_iou += iou.item()
                cnt += 1

        avg_val_iou = total_iou / max(1, cnt)
        avg_val_loss = val_loss / max(1, cnt)
        print(f"Epoch {epoch}: val_loss={avg_val_loss:.4f}, val_iou={avg_val_iou:.4f}")

        # save checkpoint
        ckpt_path = os.path.join(args.save_dir, f"epochs+{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            best_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path}")

    print("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='dataset', help='dataset folder (images/ masks/)')
    parser.add_argument('--save_dir', default='checkpoints', help='where to save checkpoints')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--encoder', default='resnet50')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    args = parser.parse_args()
    train(args)
