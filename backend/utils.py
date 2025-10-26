import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory with input images.
            mask_dir (str): Path to the directory with mask images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image
        ])
        self.mask_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

        # Get all image filenames in the directory
        self.images = [f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]

    def __len__(self):
        """Return the total number of samples"""
        return len(self.images)

    def __getitem__(self, idx):
        """Fetch a single sample: an image and its corresponding mask"""
        img_filename = self.images[idx]
        img_path = os.path.join(self.image_dir, img_filename)

        # Assume that masks have the same name but with '_mask' appended
        mask_filename = img_filename.replace('_sat.jpg', '_mask.png')
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply transforms
        image = self.transform(image)
        mask = self.mask_transform(mask)

        # Binarize the mask (if not already binarized)
        mask = (mask > 0.5).float()

        return image, mask
