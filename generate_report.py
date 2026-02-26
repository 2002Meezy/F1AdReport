"""
generate_report.py — Script principal para gerar o relatório PDF de marcas em vídeos de F1.

Uso:
    python generate_report.py --weights best.pt --source video.mp4
    python generate_report.py --weights best.pt --source video.mp4 --no-agent
    python generate_report.py --weights best.pt --source video.mp4 --output report.pdf
"""

import sys
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from brand_tracker import BrandTracker
from report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="F1 Ad Report — PDF Generation Pipeline")
    parser.add_argument('--weights', type=str, required=True, help='Path to trained YOLO weights (.pt)')
    parser.add_argument('--source', type=str, required=True, help='Path to input video file')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='IoU threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    parser.add_argument('--device', type=str, default='0', help='Device (0 for GPU, cpu for CPU)')
    parser.add_argument('--output', type=str, default=None, help='Output PDF path')
    parser.add_argument('--lm-studio-url', type=str, default='http://26.198.160.131:1234/v1',
                        help='LM Studio API URL')
    parser.add_argument('--no-agent', action='store_true', help='Skip AI analysis, generate data-only report')
    args = parser.parse_args()

    # Validate inputs
    weights = Path(args.weights)
    if not weights.exists():
        print(f"Error: Weights not found at {weights}")
        sys.exit(1)

    source = Path(args.source)
    if not source.exists():
        print(f"Error: Source video not found at {source}")
        sys.exit(1)

    # Step 1: Load model
    print(f"Loading YOLO model from {weights}...", flush=True)
    model = YOLO(weights.as_posix())

    # Step 2: Get video info
    cap = cv2.VideoCapture(source.as_posix())
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Video: {source.name} | FPS: {video_fps} | Total Frames: {total_video_frames}", flush=True)

    # Step 3: Initialize tracker
    class_names = list(model.names.values()) if hasattr(model, 'names') else []
    print(f"Tracking {len(class_names)} classes: {class_names}", flush=True)
    tracker = BrandTracker(class_names=class_names, fps=video_fps)

    # Step 4: Run inference with tracking
    print(f"\nRunning inference on {source.name}...", flush=True)
    results = model.predict(
        source=source.as_posix(),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=False,
        stream=True,
    )

    frame_count = 0
    for r in results:
        tracker.update(frame_count, r)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_video_frames} frames...", end='\r', flush=True)

    print(f"\nInference completed. {frame_count} frames processed.", flush=True)

    # Step 5: Show metrics summary
    metrics = tracker.get_metrics()
    metrics_text = tracker.get_metrics_text()
    print(f"\n{metrics_text}", flush=True)

    # Step 6: AI Analysis (optional)
    ai_analysis = None
    if not args.no_agent:
        print("\n🤖 Generating AI analysis via LM Studio...", flush=True)
        try:
            from agent import generate_analysis
            ai_analysis = generate_analysis(
                metrics_text=metrics_text,
                lm_studio_url=args.lm_studio_url
            )
            print("✅ AI analysis generated successfully.", flush=True)
        except Exception as e:
            print(f"⚠️  AI analysis failed: {e}", flush=True)
            print("Continuing with data-only report.", flush=True)
    else:
        print("\nSkipping AI analysis (--no-agent flag).", flush=True)

    # Step 7: Generate PDF
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = source.parent / f"{source.stem}_report.pdf"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n📄 Generating PDF report...", flush=True)
    generator = ReportGenerator()
    generator.generate(
        metrics=metrics,
        ai_analysis=ai_analysis,
        video_name=source.name,
        output_path=str(output_path)
    )

    print(f"✅ Report saved to: {output_path}", flush=True)
    print("\nDone!", flush=True)


if __name__ == '__main__':
    main()
