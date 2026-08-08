import argparse
import os
from ultralytics import YOLO

def train_extended_model(data_yaml, epochs, batch_size, img_size, base_model):
    print("=" * 60)
    print(" 🛠️  Training Extended PPE Detection Model (Gloves & Boots) ")
    print("=" * 60)

    if not os.path.exists(data_yaml):
        print(f"❌ Error: Dataset config file '{data_yaml}' not found!")
        return

    print(f"📦 Base Model: {base_model}")
    print(f"📄 Config File: {data_yaml}")
    print(f"🔄 Epochs: {epochs} | Batch Size: {batch_size} | Image Size: {img_size}")

    # Load YOLOv8 model
    model = YOLO(base_model)

    # Start training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        name="yolov8n_extended_ppe",
        project="output",
        exist_ok=True,
    )

    print("\n✅ Training Complete!")
    best_weights = os.path.join("output", "yolov8n_extended_ppe", "weights", "best.pt")
    if os.path.exists(best_weights):
        target_path = os.path.join("models", "best_extended.pt")
        os.makedirs("models", exist_ok=True)
        import shutil
        shutil.copy(best_weights, target_path)
        print(f"🏆 Saved new best model to: '{target_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on extended PPE dataset (Gloves, Safety Boots, etc.)")
    parser.add_argument("--data", type=str, default="data/ppe_data_extended.yaml", help="Path to extended dataset yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--base", type=str, default="models/yolov8n.pt", help="Pretrained weights base")

    args = parser.parse_args()
    train_extended_model(args.data, args.epochs, args.batch, args.imgsz, args.base)
