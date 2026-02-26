import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=str(Path.cwd() / 'configs' / 'data.yaml'))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=-1)  # -1 lets Ultralytics choose automatically
    parser.add_argument('--device', type=str, default='0')  # '0' for first GPU, or 'cpu'
    parser.add_argument('--project', type=str, default=str(Path.cwd() / 'runs' / 'train'))
    parser.add_argument('--name', type=str, default='y11s-f1ads')
    parser.add_argument('--weights', type=str, default='yolo11s.pt')
    args = parser.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_path}")

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
        patience=20,
        amp=False,
        workers=0
    )

    # Print path to best weights
    try:
        best = Path(results.save_dir) / 'weights' / 'best.pt'
        print(f"Best weights: {best}")
    except Exception:
        pass


if __name__ == '__main__':
    main()
