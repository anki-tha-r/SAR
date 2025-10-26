# # import torch
# # from unet import UNet
# # from utils import RoadDataset  # Ensure this exists and is correctly implemented
# # from torch.utils.data import DataLoader
# # from torch import nn
# # import torch.nn.functional as F
# # import os

# # # Load the trained model
# # model = UNet()
# # model.load_state_dict(torch.load('land_checkpoints/land_unet_model.pth'))
# # model.eval()  # Set the model to evaluation mode

# # # Define dataset and DataLoader
# # val_image_dir = os.path.join("dataset", "val", "images")
# # val_mask_dir = os.path.join("dataset", "val", "masks")

# # dataset = RoadDataset(val_image_dir, val_mask_dir)  # Make sure these folders have images/masks
# # data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

# # # Check if validation dataset is empty
# # if len(data_loader) == 0:
# #     print("❌ No validation data found. Please check 'dataset/val/images' and 'dataset/val/masks'.")
# #     exit()

# # # Set device
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model.to(device)

# # # Evaluation loop
# # total_loss = 0.0
# # criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss

# # # Run inference on the dataset
# # with torch.no_grad():
# #     for img, mask in data_loader:
# #         img, mask = img.to(device), mask.to(device)

# #         # Get the model's predictions
# #         pred = model(img)

# #         # Calculate loss
# #         loss = criterion(pred, mask)
# #         total_loss += loss.item()

# #     avg_loss = total_loss / len(data_loader)

# # print(f"✅ Evaluation complete! Average BCE Loss: {avg_loss:.4f}")


# # """
# # Evaluate a saved checkpoint on validation set:
# # python evaluate.py --data_dir dataset --checkpoint checkpoints/best_model.pth
# # """
# # import argparse
# # import torch
# # from torch.utils.data import DataLoader
# # import albumentations as album
# # import segmentation_models_pytorch as smp

# # from utils import list_images_and_masks, LandCoverDataset, CLASS_RGB


# # def get_validation_augmentation():
# #     return album.Compose([
# #         album.CenterCrop(height=512, width=512, always_apply=True),
# #     ])


# # def to_tensor(x, **kwargs):
# #     return x.transpose(2, 0, 1).astype('float32')


# # def get_preprocessing(preprocessing_fn=None):
# #     transforms = []
# #     if preprocessing_fn:
# #         transforms.append(album.Lambda(image=preprocessing_fn))
# #     transforms.append(album.Lambda(image=to_tensor, mask=to_tensor))
# #     return album.Compose(transforms)


# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--data_dir', default='dataset')
# #     parser.add_argument('--checkpoint', required=True)
# #     parser.add_argument('--encoder', default='resnet50')
# #     args = parser.parse_args()

# #     images, masks = list_images_and_masks(args.data_dir + '/images', args.data_dir + '/masks')
# #     assert len(images) > 0, "No images found"

# #     preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, 'imagenet')
# #     val_dataset = LandCoverDataset(images, masks, class_rgb_values=CLASS_RGB,
# #                                    augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
# #     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     model = smp.DeepLabV3Plus(encoder_name=args.encoder, encoder_weights=None, classes=len(CLASS_RGB), activation='sigmoid')
# #     model.load_state_dict(torch.load(args.checkpoint, map_location=device))
# #     model.to(device)
# #     model.eval()

# #     loss_fn = smp.utils.losses.DiceLoss()
# #     iou_metric = smp.utils.metrics.IoU(threshold=0.5)

# #     total_loss = 0.0
# #     total_iou = 0.0
# #     n = 0
# #     with torch.no_grad():
# #         for image, mask in val_loader:
# #             image = image.to(device)
# #             mask = mask.to(device)
# #             pred = model(image)
# #             total_loss += loss_fn(pred, mask).item()
# #             total_iou += iou_metric(pred, mask).item()
# #             n += 1

# #     print(f"Mean Dice Loss: {total_loss / n:.4f}")
# #     print(f"Mean IoU: {total_iou / n:.4f}")

# import os
# import torch
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from PIL import Image
# import segmentation_models_pytorch as smp
# import numpy as np
# from tqdm import tqdm

# # --- Custom Dataset for Validation ---
# class LandDataset(Dataset):
#     def __init__(self, image_dir, mask_dir, transform=None):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.transform = transform
#         self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
#         self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.image_dir, self.image_files[idx])
#         mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

#         image = Image.open(img_path).convert("RGB")
#         mask = Image.open(mask_path).convert("L")

#         if self.transform:
#             image = self.transform(image)
#             mask = self.transform(mask)

#         return image, mask

# # --- IoU (Intersection over Union) Metric ---
# def calculate_iou(pred, mask):
#     pred = pred > 0.5
#     mask = mask > 0.5

#     intersection = (pred & mask).sum().item()
#     union = (pred | mask).sum().item()
#     return intersection / union if union != 0 else 0

# # --- Evaluation Function ---
# def evaluate(model_path, dataset_root="dataset/val", batch_size=4, device="cuda" if torch.cuda.is_available() else "cpu"):
#     images_dir = os.path.join(dataset_root, "images")
#     masks_dir = os.path.join(dataset_root, "masks")

#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor()
#     ])

#     val_dataset = LandDataset(images_dir, masks_dir, transform=transform)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     model = smp.DeepLabV3Plus(
#         encoder_name='resnet50',
#         encoder_weights=None,
#         classes=1,        # or len(CLASS_RGB)
#         activation='sigmoid'
#     )
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     model.eval()


#     iou_scores = []

#     with torch.no_grad():
#         for imgs, masks in tqdm(val_loader, desc="Evaluating"):
#             imgs, masks = imgs.to(device), masks.to(device)
#             preds = torch.sigmoid(model(imgs))
#             iou = calculate_iou(preds > 0.5, masks > 0.5)
#             iou_scores.append(iou)

#     avg_iou = np.mean(iou_scores)
#     print(f"\n✅ Evaluation Complete | Mean IoU: {avg_iou:.4f}")

# if __name__ == "__main__":
#     # Example usage
#     evaluate(model_path="checkpoints/epoch_10.pth")

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp


# --- Custom Dataset (images only, no masks needed) ---
class LandTestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        return image_tensor, self.image_files[idx]


# --- Function to Save Prediction Masks ---
def save_prediction(pred_tensor, save_path):
    pred_np = pred_tensor.squeeze().cpu().numpy()
    pred_np = (pred_np > 0.5).astype(np.uint8) * 255  # threshold to binary mask
    Image.fromarray(pred_np).save(save_path)


# --- Main Evaluation / Inference Function ---
def evaluate_inference_only(
    model_path,
    input_dir="dataset/val/images",
    output_dir="outputs/val_predictions",
    encoder_name="resnet50",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    # Create output directory if missing
    os.makedirs(output_dir, exist_ok=True)

    # Transform for input images
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # match your training size
        transforms.ToTensor(),
    ])

    # Load dataset and dataloader
    test_dataset = LandTestDataset(input_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize DeepLabV3+ model
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=None,  # no pretrained weights
        classes=7,             # binary segmentation
        activation=None
    )

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"\n🚀 Running inference on {len(test_dataset)} unseen images...")

    for imgs, filenames in tqdm(test_loader, desc="Generating Masks"):
        imgs = imgs.to(device)
        preds = model(imgs)
        preds = preds.detach().cpu()
        for i in range(preds.shape[0]):
            save_path = os.path.join(output_dir, filenames[i])
            save_prediction(preds[i], save_path)


    print(f"\n✅ Done! Predicted masks saved in: {output_dir}")


if __name__ == "__main__":
    evaluate_inference_only(
        model_path="checkpoints/best_model.pth",  # update path to your .pth file
        input_dir="dataset/val/images",
        output_dir="outputs/val_predictions",
        encoder_name="resnet50"
    )

