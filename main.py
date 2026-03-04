"""
main.py — Entry point for the F1 Ad Report pipeline.

Automatically processes all videos in the 'input/' folder, runs YOLO brand
detection, generates an annotated video and a PDF report with charts,
and saves everything to the 'output/' folder.

Usage:
    python main.py --weights best.pt --no-agent
    python main.py --weights best.pt
    python main.py --weights best.pt --source input/video.mp4
"""

import sys
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from brand_tracker import BrandTracker
from pdf_report import ReportGenerator


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}


def find_input_videos(input_dir: Path) -> list[Path]:
    """Encontra todos os videos na pasta input."""
    videos = []
    if input_dir.exists():
        for f in sorted(input_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(f)
    return videos


def process_video(model, source: Path, output_dir: Path, args, class_names: list[str]):
    """Processa um video: inferencia + PDF + video anotado."""

    print(f"\n{'='*60}")
    print(f"  Processing: {source.name}")
    print(f"{'='*60}")

    # Get video info
    cap = cv2.VideoCapture(source.as_posix())
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"  FPS: {video_fps} | Frames: {total_frames}", flush=True)

    # Initialize tracker
    tracker = BrandTracker(class_names=class_names, fps=video_fps)

    # Run inference — save annotated video directly to output/
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Running inference...", flush=True)
    results = model.predict(
        source=source.as_posix(),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(output_dir),
        name=source.stem,
        exist_ok=True,
        stream=True,
    )

    frame_count = 0
    for r in results:
        tracker.update(frame_count, r)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames...", end='\r', flush=True)

    print(f"  Inference done. {frame_count} frames processed.", flush=True)

    # Get metrics
    metrics = tracker.get_metrics()
    metrics_text = tracker.get_metrics_text()
    print(f"\n{metrics_text}", flush=True)

    # AI Analysis
    ai_analysis = None
    if not args.no_agent:
        print("  Generating AI analysis via LM Studio...", flush=True)
        try:
            from ai_agent import generate_analysis
            ai_analysis = generate_analysis(
                metrics_text=metrics_text,
                lm_studio_url=args.lm_studio_url
            )
            print("  AI analysis generated.", flush=True)
        except Exception as e:
            print(f"  AI analysis failed: {e}", flush=True)
    else:
        print("  Skipping AI analysis (--no-agent).", flush=True)

    # Generate PDF in output directory
    pdf_path = output_dir / f"{source.stem}_report.pdf"

    print(f"  Generating PDF...", flush=True)
    generator = ReportGenerator()
    generator.generate(
        metrics=metrics,
        ai_analysis=ai_analysis,
        video_name=source.name,
        output_path=str(pdf_path)
    )

    # Find the saved video and move it to output root
    video_subdir = output_dir / source.stem
    if video_subdir.exists():
        for f in video_subdir.iterdir():
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
                dest = output_dir / f"{source.stem}_analyzed{f.suffix}"
                f.replace(dest)
                print(f"  Video saved: {dest.name}", flush=True)
                break
        # Clean up empty subfolder
        try:
            video_subdir.rmdir()
        except OSError:
            pass

    print(f"  PDF saved:   {pdf_path.name}", flush=True)
    print(f"  Output dir:  {output_dir}", flush=True)

    return pdf_path


def main():
    parser = argparse.ArgumentParser(description="F1 Ad Report - Video Analysis Pipeline")
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLO weights (.pt)')
    parser.add_argument('--source', type=str, default=None,
                        help='Path to a specific video (default: all videos in input/)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='IoU threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    parser.add_argument('--device', type=str, default='0', help='Device (0=GPU, cpu=CPU)')
    parser.add_argument('--input_dir', type=str, default='input', help='Input folder')
    parser.add_argument('--output_dir', type=str, default='output', help='Output folder')
    parser.add_argument('--lm-studio-url', type=str, default='http://26.198.160.131:1234/v1',
                        help='LM Studio API URL')
    parser.add_argument('--no-agent', action='store_true', help='Skip AI analysis')
    args = parser.parse_args()

    # Validate weights
    weights = Path(args.weights)
    if not weights.exists():
        print(f"Error: Weights not found at {weights}")
        sys.exit(1)

    # Load model
    print(f"Loading YOLO model from {weights}...", flush=True)
    model = YOLO(weights.as_posix())
    class_names = list(model.names.values()) if hasattr(model, 'names') else []
    print(f"Model loaded. {len(class_names)} classes.", flush=True)

    # Determine videos to process
    output_dir = Path(args.output_dir)

    if args.source:
        # Process a single video
        source = Path(args.source)
        if not source.exists():
            print(f"Error: Video not found at {source}")
            sys.exit(1)
        videos = [source]
    else:
        # Process all videos in input/
        input_dir = Path(args.input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        videos = find_input_videos(input_dir)

        if not videos:
            print(f"\nNo videos found in '{input_dir}/'.")
            print(f"Place your video files (.mp4, .avi, .mkv) in the '{input_dir}/' folder and run again.")
            sys.exit(0)

        print(f"\nFound {len(videos)} video(s) in '{input_dir}/':")
        for v in videos:
            print(f"  - {v.name}")

    # Process each video
    for video in videos:
        process_video(model, video, output_dir, args, class_names)

    print(f"\n{'='*60}")
    print(f"  All done! Results in '{output_dir}/'")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
