import os
import shutil
import random

# =========================================================
# CONFIGURATION
# =========================================================
INPUT_FOLDER = r"C:\Users\betas\OneDrive\Documentos\Elite_Grup\bogota_wro\ai_core\imagesraw"        # Folder containing your 37 raw photos
OUTPUT_FOLDER = r"C:\Users\betas\OneDrive\Documentos\Elite_Grup\bogota_wro\ai_core"   # Where the final YOLO dataset will be saved
OBJECT_NAME = "bandola_andina"         # Class name for the robot tracking system
# =========================================================

def make_full_dataset():
    # 1. Establish the clean YOLO directory structure
    train_img_path = os.path.join(OUTPUT_FOLDER, "train", "images")
    train_lbl_path = os.path.join(OUTPUT_FOLDER, "train", "labels")
    val_img_path = os.path.join(OUTPUT_FOLDER, "val", "images")
    val_lbl_path = os.path.join(OUTPUT_FOLDER, "val", "labels")

    for folder in [train_img_path, train_lbl_path, val_img_path, val_lbl_path]:
        os.makedirs(folder, exist_ok=True)

    # 2. Grab all photos
    supported_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: The input folder '{INPUT_FOLDER}' does not exist.")
        return

    images = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(supported_exts)]

    if len(images) == 0:
        print(f"Error: No images found in '{INPUT_FOLDER}'. Add your 37 photos.")
        return

    print(f"Found {len(images)} photos of the Bandola Andina. Splitting data...")

    # Shuffle for clean division
    random.seed(42)
    random.shuffle(images)

    # Split: 80% train (~29 images), 20% validation (~8 images)
    split_idx = int(len(images) * 0.8)
    train_set = images[:split_idx]
    val_set = images[split_idx:]

    # 3. Process and automatically tag the entire frame as the object
    def process_and_auto_label(image_list, target_img_dir, target_lbl_dir):
        for img_name in image_list:
            # Copy Image
            src_img = os.path.join(INPUT_FOLDER, img_name)
            shutil.copy(src_img, os.path.join(target_img_dir, img_name))

            # Auto-generate Bounding Box Label
            # Format: <class_id> <x_center> <y_center> <width> <height>
            # (0.5 0.5 1.0 1.0) creates a box covering the exact center and full bounds of the image
            base_name = os.path.splitext(img_name)[0]
            txt_name = f"{base_name}.txt"
            
            with open(os.path.join(target_lbl_dir, txt_name), 'w') as f:
                f.write("0 0.5 0.5 1.0 1.0\n")

    process_and_auto_label(train_set, train_img_path, train_lbl_path)
    process_and_auto_label(val_set, val_img_path, val_lbl_path)

    # 4. Generate the dataset configuration file (data.yaml)
    yaml_content = f"""path: {os.path.abspath(OUTPUT_FOLDER).replace(chr(92), '/')}
train: train/images
val: val/images

names:
  0: {OBJECT_NAME}
"""
    yaml_file_path = os.path.join(OUTPUT_FOLDER, "data.yaml")
    with open(yaml_file_path, "w") as f:
        f.write(yaml_content)

    print("\n=======================================================")
    print("✓ Full YOLO Dataset Setup Finished Successfully!")
    print(f"Total Photos Processed: {len(images)}")
    print(f"Training: {len(train_set)} photos -> {train_img_path}")
    print(f"Validation: {len(val_set)} photos -> {val_img_path}")
    print(f"YOLO Configuration Ready: {yaml_file_path}")
    print("=======================================================")

if __name__ == "__main__":
    make_full_dataset()
