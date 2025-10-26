# import os
# import shutil

# def segregate_images(dataset_dir):
#     """
#     Moves images inside `dataset_dir` into two subfolders:
#     - images/: contains files with '_sat' in their name
#     - masks/: contains files with '_mask' in their name
#     """

#     # Create subdirectories if not already present
#     images_dir = os.path.join(dataset_dir, 'images')
#     masks_dir = os.path.join(dataset_dir, 'masks')
#     os.makedirs(images_dir, exist_ok=True)
#     os.makedirs(masks_dir, exist_ok=True)

#     # Loop through all files in the dataset directory
#     for filename in os.listdir(dataset_dir):
#         file_path = os.path.join(dataset_dir, filename)

#         # Skip directories
#         if os.path.isdir(file_path):
#             continue

#         # Move files based on name pattern
#         if '_sat' in filename:
#             shutil.move(file_path, os.path.join(images_dir, filename))
#         elif '_mask' in filename:
#             shutil.move(file_path, os.path.join(masks_dir, filename))

#     print("✅ Segregation complete.")
#     print(f" - Satellite images moved to: {images_dir}")
#     print(f" - Mask images moved to: {masks_dir}")

# if __name__ == "__main__":
#     dataset_folder = "dataset/val"  # Change path if needed
#     segregate_images(dataset_folder)

import os

def rename_files_in_folder(folder_path):
    """
    Renames all files in the given folder by removing '_sat' and '_mask'
    from their filenames. Keeps the original file extensions.
    """
    renamed_count = 0

    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)

        # Skip directories
        if os.path.isdir(old_path):
            continue

        # Split name and extension
        name, ext = os.path.splitext(filename)

        # Remove unwanted parts
        new_name = name.replace('_sat', '').replace('_mask', '') + ext

        # If different, rename
        if new_name != filename:
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            renamed_count += 1
            print(f"Renamed: {filename} → {new_name}")

    print(f"\n✅ Done. Renamed {renamed_count} files in '{folder_path}'.")


if __name__ == "__main__":
    for sub in ['dataset/images', 'dataset/masks', 'dataset/val/images', 'dataset/val/masks']:
        rename_files_in_folder(sub)


