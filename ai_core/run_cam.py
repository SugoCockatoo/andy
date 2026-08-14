import os
import sys
import cv2
from ultralytics import YOLO

# =========================================================
# CONFIGURATION
# =========================================================
TENSORRT_ENGINE = "best.engine"
PYTORCH_WEIGHTS = "best.pt"
CAMERA_SOURCE = 0  # Change to 0, 1, or your CSI camera string
# =========================================================

def run_live_window_tracker():
    # 1. Compilation fail-safe check
    if not os.path.exists(TENSORRT_ENGINE):
        if not os.path.exists(PYTORCH_WEIGHTS):
            print(f"❌ Error: Missing '{PYTORCH_WEIGHTS}'. Copy it here first.")
            sys.exit(1)
        print("⚙️ Compiling TensorRT Engine (First-time setup takes a few minutes)...")
        pt_model = YOLO(PYTORCH_WEIGHTS)
        pt_model.export(format="engine", device=0, half=True)

    # 2. Load the optimized engine
    print("🚀 Loading tracking module...")
    model = YOLO(TENSORRT_ENGINE, task="detect")

    # 3. Open Video Stream via OpenCV
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera source {CAMERA_SOURCE}")
        sys.exit(1)

    print("\n🎥 Tracking Active. Press 'q' inside the video window to exit.")
    print("=======================================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame from camera source.")
            break

        # Get frame properties for spatial math
        height, width, _ = frame.shape
        camera_center_x = int(width / 2)

        # Draw a visual reference line down the exact center of the screen (Target Crosshair)
        cv2.line(frame, (camera_center_x, 0), (camera_center_x, height), (255, 255, 255), 1)

        # Run inference on a single frame (device=0 forces Jetson GPU)
        results = model.predict(source=frame, device=0, verbose=False)
        
        # Default action when no object is found
        action_text = "SEARCHING (No Bandola Detected)"
        text_color = (0, 0, 255) # Red text

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 4. Extract pixel coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = box.conf[0].item()

                # Calculate object center point
                bandola_center_x = int((x1 + x2) / 2)
                bandola_center_y = int((y1 + y2) / 2)

                # Error calculation for the motor controller
                error_x = bandola_center_x - camera_center_x

                # Determine tracking directions
                if abs(error_x) < 25:  # 25-pixel center tolerance deadzone
                    action_text = "HOLD POSITION (Centered)"
                    text_color = (0, 255, 0)  # Green text
                elif error_x > 0:
                    action_text = f"TURN RIGHT (Offset: +{abs(error_x)}px)"
                    text_color = (0, 165, 255) # Orange text
                else:
                    action_text = f"TURN LEFT (Offset: -{abs(error_x)}px)"
                    text_color = (0, 165, 255) # Orange text

                # 5. Visual Rendering Overlays
                # Draw the bounding box (Bright Green Outline)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Draw a dot right at the center of the bandola
                cv2.circle(frame, (bandola_center_x, bandola_center_y), 7, (0, 255, 0), -1)

                # Draw a line connecting the camera center directly to the bandola center
                cv2.line(frame, (camera_center_x, bandola_center_y), (bandola_center_x, bandola_center_y), (255, 255, 0), 2)

                # Draw class label string and confidence rating above the frame box
                label = f"Bandola: {confidence*100:.1f}%"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # We break here to only process the strongest primary target per frame
                break

        # 6. Render the live status onto the screen
        # Displays the current motor adjustment command in the top-left corner
        cv2.putText(frame, f"Robot Action: {action_text}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

        # Open and update the graphical display window on the Jetson desktop
        cv2.imshow("WRO Robot - Bandola Andina Tracker", frame)

        # Break loop instantly if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up operations upon exiting
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Live window loop terminated cleanly.")

if __name__ == "__main__":
    run_live_window_tracker()
