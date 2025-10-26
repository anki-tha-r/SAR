# import os
# from torch.utils.data import Dataset
# from PIL import Image
# import torchvision.transforms as T

# class LandDataset(Dataset):
#     def __init__(self, image_dir, mask_dir, transform=None):
#         """
#         Args:
#             image_dir (str): Path to the directory with input images.
#             mask_dir (str): Path to the directory with mask images.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.transform = transform or T.Compose([
#             T.Resize((256, 256)),
#             T.ToTensor(),
#             T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize image
#         ])
#         self.mask_transform = T.Compose([
#             T.Resize((256, 256)),
#             T.ToTensor()
#         ])

#         # Get all image filenames in the directory
#         self.images = [f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]

#     def __len__(self):
#         """Return the total number of samples"""
#         return len(self.images)

#     def __getitem__(self, idx):
#         """Fetch a single sample: an image and its corresponding mask"""
#         img_filename = self.images[idx]
#         img_path = os.path.join(self.image_dir, img_filename)

#         # Assume that masks have the same name but with '_mask' appended
#         mask_filename = img_filename.replace('_sat.jpg', '_mask.png')
#         mask_path = os.path.join(self.mask_dir, mask_filename)

#         image = Image.open(img_path).convert("RGB")
#         mask = Image.open(mask_path).convert("L")

#         # Apply transforms
#         image = self.transform(image)
#         mask = self.mask_transform(mask)

#         # Binarize the mask (if not already binarized)
#         mask = (mask > 0.5).float()

#         return image, mask


import os
import cv2
import numpy as np
from torch.utils.data import Dataset

# Default DeepGlobe classes (RGB order) — change if your dataset is different
CLASS_NAMES = ['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown']
CLASS_RGB = np.array([
    [0, 255, 255],   # urban_land
    [255, 255, 0],   # agriculture_land
    [255, 0, 255],   # rangeland
    [0, 255, 0],     # forest_land
    [0, 0, 255],     # water
    [255, 255, 255], # barren_land
    [0, 0, 0]        # unknown
], dtype=np.uint8)


def one_hot_encode(label, label_values=CLASS_RGB):
    """
    Convert an HxWx3 RGB mask to an HxWxC one-hot array (dtype uint8).
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype('uint8')
    return semantic_map


def reverse_one_hot(image_onehot):
    """
    From HxWxC one-hot -> HxW integer class map
    """
    return np.argmax(image_onehot, axis=-1)


def colour_code_segmentation(class_map, label_values=CLASS_RGB):
    """
    Map HxW integer class map to HxWx3 color-coded mask
    """
    colour_codes = np.array(label_values)
    return colour_codes[class_map.astype(int)]


class LandCoverDataset(Dataset):
    """
    Dataset returning (image, mask) pairs.
    image: HxWx3 uint8 (or preprocessed CHW float32 if preprocessing provided)
    mask: HxWxC one-hot (or CHW if preprocessing applied)
    """

    def __init__(self, images_list, masks_list, class_rgb_values=None, augmentation=None, preprocessing=None):
        assert len(images_list) == len(masks_list), "Images and masks lists must be same length"
        self.image_paths = images_list
        self.mask_paths = masks_list
        self.class_rgb_values = class_rgb_values if class_rgb_values is not None else CLASS_RGB
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[idx]), cv2.COLOR_BGR2RGB)

        mask = one_hot_encode(mask, self.class_rgb_values).astype('float32')

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask


def list_images_and_masks(images_dir, masks_dir):
    """
    Match images and masks by filename (without extension). Returns two parallel lists.
    """
    images = {}
    for fname in os.listdir(images_dir):
        if fname.startswith('.'):
            continue
        images[os.path.splitext(fname)[0]] = os.path.join(images_dir, fname)

    masks = {}
    for fname in os.listdir(masks_dir):
        if fname.startswith('.'):
            continue
        masks[os.path.splitext(fname)[0]] = os.path.join(masks_dir, fname)

    common = sorted(set(images.keys()).intersection(set(masks.keys())))
    images_list = [images[k] for k in common]
    masks_list = [masks[k] for k in common]
    return images_list, masks_list
