import os
import random
import shutil

def create_val_split(train_dir='dataset/', val_dir='dataset/val', split_ratio=0.1):
    train_images = os.path.join(train_dir, 'images')
    train_masks = os.path.join(train_dir, 'masks')
    val_images = os.path.join(val_dir, 'images')
    val_masks = os.path.join(val_dir, 'masks')

    os.makedirs(val_images, exist_ok=True)
    os.makedirs(val_masks, exist_ok=True)

    image_files = os.listdir(train_images)
    val_files = random.sample(image_files, int(len(image_files) * split_ratio))

    for file in val_files:
        shutil.move(os.path.join(train_images, file), os.path.join(val_images, file))
        shutil.move(os.path.join(train_masks, file), os.path.join(val_masks, file))

    print(f"Moved {len(val_files)} images to validation set.")

if __name__ == "__main__":
    create_val_split()
