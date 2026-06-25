import os
import shutil
import glob

def combine_yolo_with_bdd100k(bdd_root, custom_root, target_split='train2017', file_prefix='custom_'):
    """
    Combines a custom YOLO dataset with the BDD100K dataset.

    Args:
        bdd_root (str): The root path of the BDD100K dataset.
        custom_root (str): The root path of the custom YOLO dataset.
        target_split (str): The BDD100K split to add data to ('train2017' or 'val2017').
        file_prefix (str): A prefix to add to custom files to avoid name collisions.
    """
    # --- 1. Define Source and Destination Paths ---
    # Source paths for your custom data
    custom_images_src = os.path.join(custom_root, 'images')
    custom_labels_src = os.path.join(custom_root, 'labels')

    # Destination paths within the BDD100K structure
    bdd_images_dest = os.path.join(bdd_root, 'images', target_split)
    bdd_labels_dest = os.path.join(bdd_root, 'detection-object', 'labels', target_split)

    # --- 2. Safety Checks and Directory Creation ---
    if not os.path.isdir(custom_images_src) or not os.path.isdir(custom_labels_src):
        print(f"❌ Error: Your custom dataset at '{custom_root}' must contain 'images' and 'labels' subfolders.")
        return

    os.makedirs(bdd_images_dest, exist_ok=True)
    os.makedirs(bdd_labels_dest, exist_ok=True)
    
    print(f"🚀 Starting to combine custom data into BDD100K '{target_split}' split...")

    # --- 3. Find and Copy Files ---
    # Get a list of all image files from the custom dataset
    image_files = glob.glob(os.path.join(custom_images_src, '*.jpg')) + \
                  glob.glob(os.path.join(custom_images_src, '*.png'))

    if not image_files:
        print("⚠️ Warning: No images found in the custom dataset's images folder.")
        return

    copied_count = 0
    # Loop through each image and copy it and its corresponding label
    for img_path in image_files:
        base_filename = os.path.basename(img_path)
        name_part, extension = os.path.splitext(base_filename)
        
        # Define the source path for the corresponding label file
        label_filename = f"{name_part}.txt"
        label_path_src = os.path.join(custom_labels_src, label_filename)

        # Check if the corresponding label file exists before processing
        if os.path.exists(label_path_src):
            # Add the prefix to the new filenames to avoid overwriting BDD100k files
            new_image_filename = f"{file_prefix}{base_filename}"
            new_label_filename = f"{file_prefix}{label_filename}"

            # Define the final destination paths
            dest_img_path = os.path.join(bdd_images_dest, new_image_filename)
            dest_label_path = os.path.join(bdd_labels_dest, new_label_filename)

            # Copy the image and label files
            shutil.copy2(img_path, dest_img_path)
            shutil.copy2(label_path_src, dest_label_path)
            copied_count += 1
        else:
            print(f"  - ⚠️ Warning: Label file not found for '{base_filename}'. Skipping.")

    print("\n✅ Combination complete!")
    print(f"Total custom images and labels copied: {copied_count}")


# --- Main execution block ---
if __name__ == '__main__':
    # ========================== USER CONFIGURATION ==========================
    # 1. Set the root path of your BDD100K dataset
    bdd100k_dataset_root = 'D:/Shehsaath/dataset/bdd100k_dataset'

    # 2. Set the root path of your custom YOLO dataset
    #    (This folder must contain 'images' and 'labels' subfolders)
    custom_yolo_dataset_root = 'D:/Shehsaath/final_dataset_v3'

    # 3. Choose which BDD100K split to add the custom data to
    #    Options: 'train2017' or 'val2017'
    target_split_name = 'train2017'
    # ========================================================================

    # --- SAFETY CHECK ---
    # A simple check to prevent running the script with default placeholder paths
    if '/path/to/your/' in bdd100k_dataset_root or '/path/to/your/' in custom_yolo_dataset_root:
        print("🚨 Please update the placeholder paths in the 'USER CONFIGURATION' section before running!")
    else:
        combine_yolo_with_bdd100k(bdd100k_dataset_root, custom_yolo_dataset_root, target_split_name)