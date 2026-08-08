import argparse
import os
import time
import cv2
import torch
from ultralytics import YOLO

def run_detector(source, weights, conf_thresh, save_output, show_window):
    # Verify model weights file
    if not os.path.exists(weights):
        print(f"❌ Error: Weights file not found at '{weights}'")
        return

    print(f"🚀 Loading YOLOv8 model from '{weights}'...")
    model = YOLO(weights)
    names = model.names

    # Non-compliance classes for alert highlights (Head, Face, Torso, Hands, Feet)
    violation_classes = {
        "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
        "NO-Gloves", "NO-Safety Shoes", "NO-Boots"
    }

    # Determine input source type (webcam integer vs string file/stream)
    is_webcam = str(source).isdigit()
    source_val = int(source) if is_webcam else source

    print(f"🎥 Connecting to source: '{source}'")
    cap = cv2.VideoCapture(source_val)

    if not cap.isOpened():
        print(f"❌ Error: Unable to open video source '{source}'")
        return

    # Video Writer setup if saving output stream
    writer = None
    if save_output:
        os.makedirs("output", exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        save_path = os.path.join("output", "live_detection_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        print(f"💾 Saving processed video stream to: '{save_path}'")

    print("\n🟢 Monitoring started! Press 'q' in the window to stop.\n")

    prev_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of stream or cannot fetch frame.")
                break

            # Perform object detection inference
            results = model.predict(source=frame, conf=conf_thresh, verbose=False)
            res = results[0]

            violations = []
            boxes = res.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = names[cls_id]
                conf = float(box.conf[0].item())
                if cls_name in violation_classes:
                    violations.append(f"{cls_name} ({conf:.0%})")

            # Draw annotations (bounding boxes & labels)
            annotated_frame = res.plot()

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            # Onscreen FPS Display
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Onscreen Safety Alert Banner if violation is detected
            if violations:
                alert_text = f"⚠️ SAFETY VIOLATION DETECTED: {', '.join(set(violations))}"
                # Red banner background
                cv2.rectangle(
                    annotated_frame,
                    (0, annotated_frame.shape[0] - 50),
                    (annotated_frame.shape[1], annotated_frame.shape[0]),
                    (0, 0, 200),
                    -1,
                )
                cv2.putText(
                    annotated_frame,
                    alert_text,
                    (20, annotated_frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Write frame if saving output
            if writer is not None:
                writer.write(annotated_frame)

            # Show GUI Window
            if show_window:
                cv2.imshow("Construction Site PPE Detection - Live Feed", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("User requested exit.")
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("🛑 Stream closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live PPE Detection for Webcam, CCTV RTSP stream, or Video/Image files")
    parser.add_argument("--source", type=str, default="0", help="0 for Webcam, RTSP URL (rtsp://...), video path (.mp4), or image path (.jpg)")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to trained YOLOv8 weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--save", action="store_true", help="Save detection video/image output to 'output/' directory")
    parser.add_argument("--no-show", action="store_true", help="Disable display window (for headless execution)")

    args = parser.parse_args()

    # Image prediction mode if source is a single image file
    ext = os.path.splitext(args.source)[1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        print(f"🖼️ Running inference on image: '{args.source}'")
        model = YOLO(args.weights)
        res = model.predict(source=args.source, conf=args.conf, save=args.save, project="output", name="predictions")[0]
        out_img = res.plot()
        if not args.no_show:
            cv2.imshow("PPE Image Detection", out_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print("Done.")
    else:
        run_detector(
            source=args.source,
            weights=args.weights,
            conf_thresh=args.conf,
            save_output=args.save,
            show_window=not args.no_show,
        )
