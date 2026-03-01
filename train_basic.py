"""
train_basic.py — Basic YOLO training for F1 brand detection.

Trains a YOLO11 model on the F1 brand dataset with default parameters.
Use this for a quick first training run. For improved accuracy, use
train_advanced.py instead.

Usage:
    python train_basic.py
    python train_basic.py --epochs 100 --data configs/data.yaml
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="F1 Brand Detection — Basic Training")
    parser.add_argument('--data', type=str, default=str(Path.cwd() / 'configs' / 'data.yaml'),
                        help='Path to the dataset YAML config')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--batch', type=int, default=-1,
                        help='Batch size (-1 for auto)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device: "0" for GPU, "cpu" for CPU')
    parser.add_argument('--project', type=str, default=str(Path.cwd() / 'runs' / 'train'),
                        help='Output directory for training runs')
    parser.add_argument('--name', type=str, default='y11s-f1ads',
                        help='Name for this training run')
    parser.add_argument('--weights', type=str, default='yolo11s.pt',
                        help='Pre-trained weights to start from')
    args = parser.parse_args()

    # Validate dataset config
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_path}")

    # Load pre-trained model and start training
    model = YOLO(args.weights)
    results = model.train(
        data=data_path.as_posix(),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=None if args.batch == -1 else args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=True,
        seed=42,
        patience=20,      # Early stopping after 20 epochs without improvement
        amp=False,
        workers=0
    )

    # Print path to best weights
    try:
        best = Path(results.save_dir) / 'weights' / 'best.pt'
        print(f"\nTraining complete! Best weights: {best}")
    except Exception:
        pass


if __name__ == '__main__':
    main()
