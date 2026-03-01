"""
detect.py — Standalone YOLO detection on a video file.

Runs the trained YOLO model on a video, saves an annotated output video with
bounding boxes drawn on each frame, and optionally generates a PDF report.

This is useful when you want just the annotated video without the full
pipeline from main.py.

Usage:
    python detect.py --weights best.pt --source video.mp4
    python detect.py --weights best.pt --source video.mp4 --report --no_agent
"""

import sys
import argparse
from pathlib import Path
import shutil
import cv2

from brand_tracker import BrandTracker

from ultralytics import YOLO

def main():
    print("Parsing arguments...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to trained weights .pt')
    parser.add_argument('--source', type=str, required=True, help='Path to input video file')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.7)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--project', type=str, default=str(Path.cwd() / 'output'))
    parser.add_argument('--name', type=str, default='f1ads-vid')
    parser.add_argument('--save_txt', action='store_true', help='Save YOLO-format predictions')
    parser.add_argument('--same_dir', action='store_true', help='Save output video in the same directory as source')
    parser.add_argument('--show', action='store_true', help='Show results in a window')
    parser.add_argument('--nosave', action='store_true', help='Do not save output video/images')
    parser.add_argument('--report', action='store_true', help='Generate PDF report after inference')
    parser.add_argument('--lm_studio_url', type=str, default='http://26.198.160.131:1234/v1',
                        help='LM Studio API URL')
    parser.add_argument('--no_agent', action='store_true', help='Generate report without AI agent')
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"Error: Weights not found at {weights}")
        sys.exit(1)

    source = Path(args.source)
    if not source.exists():
        print(f"Error: Source video not found at {source}")
        sys.exit(1)

    print(f"Loading model from {weights}...", flush=True)
    model = YOLO(weights.as_posix())

    # Get video FPS for time calculations
    cap = cv2.VideoCapture(source.as_posix())
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    print(f"Video FPS: {video_fps}", flush=True)

    # Initialize BrandTracker with class names from the model
    class_names = list(model.names.values()) if hasattr(model, 'names') else []
    print(f"Tracking classes: {class_names}", flush=True)
    tracker = BrandTracker(class_names=class_names, fps=video_fps)

    print(f"Starting inference on {source}...", flush=True)
    print("Using stream=True to handle large video files.", flush=True)
    if args.nosave:
        print("Saving disabled for better performance.", flush=True)

    # Use stream=True to handle large videos without running out of memory
    results = model.predict(
        source=source.as_posix(),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        save=not args.nosave,
        save_txt=args.save_txt,
        vid_stride=1,
        show=args.show,
        stream=True
    )

    save_dir = None
    
    # Process results generator
    frame_count = 0
    print("Processing frames...", flush=True)
    for r in results:
        # Feed each frame's results to the BrandTracker
        tracker.update(frame_count, r)
        frame_count += 1
        if save_dir is None and not args.nosave:
            save_dir = Path(r.save_dir)
            print(f"Predictions saving to: {save_dir}", flush=True)
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...", end='\r', flush=True)
    
    print(f"\nInference completed. Total frames processed: {frame_count}", flush=True)

    # Print tracking metrics summary
    print("\n" + tracker.get_metrics_text(), flush=True)

    # Generate PDF report if requested
    if args.report:
        try:
            from pdf_report import ReportGenerator
            from ai_agent import generate_analysis

            metrics = tracker.get_metrics()
            
            ai_analysis = None
            if not args.no_agent:
                print("\nGenerating AI analysis...", flush=True)
                try:
                    ai_analysis = generate_analysis(
                        metrics_text=tracker.get_metrics_text(),
                        lm_studio_url=args.lm_studio_url
                    )
                    print("AI analysis generated successfully.", flush=True)
                except Exception as e:
                    print(f"Warning: AI analysis failed: {e}", flush=True)
                    print("Generating report without AI analysis.", flush=True)

            report_dir = Path(args.project)
            report_dir.mkdir(parents=True, exist_ok=True)
            output_pdf = report_dir / f"{source.stem}_report.pdf"

            generator = ReportGenerator()
            generator.generate(
                metrics=metrics,
                ai_analysis=ai_analysis,
                video_name=source.name,
                output_path=str(output_pdf)
            )
            print(f"\n📄 PDF Report saved to: {output_pdf}", flush=True)
        except Exception as e:
            print(f"Error generating report: {e}", flush=True)

    if not args.nosave and save_dir and args.same_dir and source.is_file():
        found_video = None
        if save_dir.exists():
            for cand in save_dir.iterdir():
                if cand.is_file() and cand.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov']:
                    if cand.stem.startswith(source.stem):
                         found_video = cand
                         break
        
        if found_video:
            dest = source.parent / f"{source.stem}_analyzed_v52{found_video.suffix}"
            try:
                print(f"Copying output video to {dest}...", flush=True)
                shutil.copy2(found_video, dest)
                print(f"Success! Output video available at: {dest}", flush=True)
            except Exception as e:
                print(f"Error copying video: {e}", flush=True)
        else:
            print(f"Warning: Could not locate output video in {save_dir} to copy to source directory.", flush=True)

if __name__ == '__main__':
    main()
