import os
import glob
from ultralytics import YOLO
import cv2
import shutil

def run_inference():
    model_path = os.path.join("models", "best.pt")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading trained YOLOv8 model from {model_path}...")
    model = YOLO(model_path)

    source_dir = "source_files"
    output_dir = "demo_results"
    os.makedirs(output_dir, exist_ok=True)

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(source_dir, ext)))

    print(f"Found {len(image_files)} sample image(s) in '{source_dir}'\n")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"--- Running Inference on: {filename} ---")
        
        # Perform prediction
        results = model.predict(source=img_path, conf=0.25, save=False)
        res = results[0]
        
        # Print detected objects
        boxes = res.boxes
        if len(boxes) == 0:
            print("  No objects detected.")
        else:
            names = res.names
            counts = {}
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = names[cls_id]
                conf = float(box.conf[0].item())
                counts[cls_name] = counts.get(cls_name, 0) + 1
                print(f"  - Detected: {cls_name} (Confidence: {conf:.2%})")
            
            print("  Summary of Detections:")
            for item, count in counts.items():
                print(f"    • {item}: {count}")

        # Save annotated image
        annotated_img = res.plot()
        out_path = os.path.join(output_dir, f"detected_{filename}")
        cv2.imwrite(out_path, annotated_img)
        print(f"  Saved detection visualization to: {out_path}\n")

if __name__ == "__main__":
    run_inference()
