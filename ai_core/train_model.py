import os
from ultralytics import YOLO

# =========================================================
# CONFIGURATION
# =========================================================
# Point this to the exact data.yaml file created by your previous script
DATA_YAML_PATH = r"C:\Users\betas\OneDrive\Documentos\Elite_Grup\bogota_wro\ai_core\data.yaml"

# Training Settings
EPOCHS = 100         # Passes through the dataset
IMAGE_SIZE = 640     # Resolution the model trains on
BATCH_SIZE = 16      # CPU training handles batch sizes well, keep at 16 or drop to 8 if RAM fills up

# CHANGED: Switched from 0 (GPU) to 'cpu'
DEVICE = 'cpu'       
# =========================================================

def train_bandola_tracker():
    # 1. Verify that your dataset configuration exists
    if not os.path.exists(DATA_YAML_PATH):
        print(f"Error: Could not find your dataset configuration file at: {DATA_YAML_PATH}")
        print("Please check your file path or make sure you ran the generator script first.")
        return

    print("=======================================================")
    print("🚀 Initializing YOLOv11 Nano Training Loop on CPU...")
    print(f"Target Dataset: {DATA_YAML_PATH}")
    print("=======================================================")

    # 2. Load the pre-trained YOLOv11 nano model from your local path
    model = YOLO(r"C:\Users\betas\OneDrive\Documentos\Elite_Grup\bogota_wro\ai_core\yolo11n.pt")

    # 3. Launch the automated training cycle
    model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=0,                  # CHANGED: Set to 0 to prevent multi-processing bugs on Windows CPU execution
        amp=False,                  # CHANGED: Disabled AMP because CPU training does not utilize GPU float16 math
        save=True,           
        project="bandola_tracking", 
        name="yolov11n_run"         
    )

    # 4. Success summary output
    print("\n=======================================================")
    print("✓ Training Cycle Complete!")
    print("=======================================================")
    print("Your trained model weights are saved at:")
    print("📂 bandola_tracking/yolov11n_run/weights/")
    print("\nNext step: Copy 'best.pt' to your Jetson Nano to track the bandola in real-time.")
    print("=======================================================")

if __name__ == "__main__":
    train_bandola_tracker()
