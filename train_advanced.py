"""
train_advanced.py — Advanced YOLO training with aggressive data augmentation.

Builds on train_basic.py with techniques to maximize accuracy on a small dataset:
- Aggressive augmentation: mixup, cutmix, copy-paste, random erasing
- Higher image resolution (960px) for detecting smaller logos
- Cosine learning rate scheduling for smoother convergence
- 3x classification loss weight to prioritize brand identification
- Dropout regularization to prevent overfitting

Usage:
    python train_advanced.py
    python train_advanced.py --epochs 200 --imgsz 1280
    python train_advanced.py --weights yolo11m.pt --freeze 10
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="F1 Ad Detection — Advanced Training v2")
    parser.add_argument('--data', type=str, default=str(Path.cwd() / 'configs' / 'data2.yaml'))
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--batch', type=int, default=-1)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--project', type=str, default=str(Path.cwd() / 'runs' / 'train'))
    parser.add_argument('--name', type=str, default='y11-f1ads-v3')
    parser.add_argument('--weights', type=str, default='yolo11s.pt',
                        help='Base weights (yolo11s.pt or yolo11m.pt)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--freeze', type=int, default=0,
                        help='Number of backbone layers to freeze (0=none, 10=freeze backbone)')
    args = parser.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_path}")

    # Load model — either fresh or from checkpoint
    if args.resume:
        print(f"Resuming training from {args.resume}...")
        model = YOLO(args.resume)
    else:
        print(f"Loading base model: {args.weights}")
        model = YOLO(args.weights)

    # Freeze layers if specified (useful for transfer learning with small datasets)
    freeze_layers = list(range(args.freeze)) if args.freeze > 0 else None

    print(f"\n{'='*60}")
    print(f"  F1 Ad Detection — Advanced Training v2")
    print(f"{'='*60}")
    print(f"  Data:       {data_path}")
    print(f"  Model:      {args.weights}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Image Size: {args.imgsz}")
    print(f"  Freeze:     {args.freeze} layers")
    print(f"{'='*60}\n")

    results = model.train(
        data=data_path.as_posix(),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=True,
        seed=42,

        # ===== PATIENCE =====
        patience=40,               # 20 → 40: more time to converge

        # ===== LEARNING RATE =====
        cos_lr=True,               # Cosine annealing LR schedule
        lr0=0.01,                  # Initial learning rate
        lrf=0.001,                 # Final LR factor (smoother decay)
        warmup_epochs=5.0,         # 3 → 5: longer warmup for stability
        warmup_momentum=0.8,

        # ===== LOSS WEIGHTS =====
        cls=1.5,                   # 0.5 → 1.5: TRIPLE classification weight
        box=7.5,                   # Keep box loss weight
        dfl=1.5,                   # Keep DFL loss weight

        # ===== REGULARIZATION =====
        dropout=0.1,               # 0.0 → 0.1: prevent overfitting
        weight_decay=0.0005,

        # ===== DATA AUGMENTATION =====
        mosaic=1.0,                # Keep mosaic (already good)
        close_mosaic=25,           # 10 → 25: mosaic on for more epochs
        mixup=0.15,                # 0.0 → 0.15: blend images for regularization
        cutmix=0.15,               # 0.0 → 0.15: cut-paste regions
        copy_paste=0.3,            # 0.0 → 0.3: paste brands into different backgrounds
        copy_paste_mode='flip',

        # ===== GEOMETRIC AUGMENTATION =====
        degrees=10.0,              # 0.0 → 10.0: slight rotation
        translate=0.2,             # 0.1 → 0.2: more position variation
        scale=0.9,                 # 0.5 → 0.9: wider scale range
        shear=2.0,                 # 0.0 → 2.0: slight shear
        perspective=0.0001,        # Subtle perspective transform
        flipud=0.0,                # No vertical flip (text would be unreadable)
        fliplr=0.5,                # Horizontal flip OK

        # ===== COLOR AUGMENTATION =====
        hsv_h=0.02,               # Slight hue variation
        hsv_s=0.7,                # Saturation variation
        hsv_v=0.4,                # Brightness variation
        erasing=0.4,              # Random erasing for robustness

        # ===== OTHER =====
        amp=False,
        workers=0,
        freeze=freeze_layers,
        multi_scale=False,          # Disabled — causes OOM on 8GB VRAM with large imgsz

        # ===== AUTO AUGMENT =====
        auto_augment='randaugment',
    )

    # Print results
    try:
        best = Path(results.save_dir) / 'weights' / 'best.pt'
        print(f"\n{'='*60}")
        print(f"  Training Complete!")
        print(f"  Best weights: {best}")
        print(f"{'='*60}")
    except Exception:
        pass


if __name__ == '__main__':
    main()
