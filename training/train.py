"""Fine-tune YOLOv8n on the labelled mosquito dataset and export it for the server.

    python training/train.py                   # after prepare_dataset.py
    python training/train.py --epochs 150 --device 0

Writes models/yolov8n_mosquito.onnx, which the server loads with DETECTOR=yolo.
"""
import argparse
import os
import shutil

from ultralytics import YOLO

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TRAINING_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(TRAINING_DIR, "data", "dataset", "data.yaml"))
    parser.add_argument("--base", default="yolov8n.pt", help="starting weights (COCO yolov8n)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="e.g. 0 for the first GPU, cpu for CPU")
    parser.add_argument("--output", default=os.path.join(REPO_ROOT, "models", "yolov8n_mosquito.onnx"))
    args = parser.parse_args()

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        project=os.path.join(TRAINING_DIR, "runs"),
        name="mosquito",
        patience=20,  # stop early once validation stops improving
    )
    best_weights = model.trainer.best
    print(f"Best weights: {best_weights}")

    # Same export settings the server's inference engine expects.
    exported = YOLO(best_weights).export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    shutil.copy(exported, args.output)
    print(f"ONNX model written to {args.output}")
    print("Check precision/recall above, then set DETECTOR=yolo and a new MODEL_VERSION in .env.")


if __name__ == "__main__":
    main()
