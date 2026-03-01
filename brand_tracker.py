"""
brand_tracker.py — Tracks brand detections across video frames.

Collects per-frame data from YOLO detections and computes metrics such as:
- Display time (seconds each brand is visible)
- Detection frequency (how many frames/detections per brand)
- Average confidence score per brand
- Overall summary text for AI analysis and PDF reports
"""

from collections import defaultdict


class BrandTracker:
    """
    Tracks brand visibility metrics across all frames of a video.

    For each frame processed, it records which brands were detected
    and their confidence scores. After processing, it can generate
    a complete metrics summary.

    Attributes:
        class_names: List of brand class names from the YOLO model.
        fps: Video frames per second (used to convert frames → seconds).
    """

    def __init__(self, class_names: list[str], fps: float = 30.0):
        """
        Initialize the tracker.

        Args:
            class_names: List of brand names matching YOLO class IDs.
            fps: Video FPS used to convert frame counts into seconds.
        """
        self.class_names = class_names
        self.fps = fps

        # Per-brand tracking data
        self.brand_frames = defaultdict(set)       # brand → set of frame indices
        self.brand_confidences = defaultdict(list)  # brand → list of confidence scores
        self.brand_detections = defaultdict(int)     # brand → total detection count

        # Overall tracking
        self.total_frames = 0

    def update(self, frame_idx: int, result):
        """
        Process a single frame's YOLO detection results.

        Args:
            frame_idx: Current frame number (0-indexed).
            result: A single YOLO result object containing boxes, classes, and confidences.
        """
        self.total_frames = max(self.total_frames, frame_idx + 1)

        if result.boxes is None or len(result.boxes) == 0:
            return

        for i in range(len(result.boxes)):
            cls_id = int(result.boxes.cls[i].item())
            confidence = float(result.boxes.conf[i].item())

            if cls_id < len(self.class_names):
                brand = self.class_names[cls_id]
                self.brand_frames[brand].add(frame_idx)
                self.brand_confidences[brand].append(confidence)
                self.brand_detections[brand] += 1

    def get_metrics(self) -> dict:
        """
        Compute all brand visibility metrics.

        Returns:
            Dictionary containing:
            - video_info: FPS, total frames, duration
            - brands: Per-brand metrics (time, frames, detections, confidence)
            - most_frequent_brand: Brand with the most frames visible
            - total_brands_detected: Number of distinct brands found
        """
        duration = self.total_frames / self.fps if self.fps > 0 else 0

        brands = {}
        for brand in sorted(self.brand_frames.keys()):
            frames_visible = len(self.brand_frames[brand])
            time_visible = round(frames_visible / self.fps, 2) if self.fps > 0 else 0
            confs = self.brand_confidences[brand]
            avg_conf = round(sum(confs) / len(confs), 4) if confs else 0

            brands[brand] = {
                "frames_visible": frames_visible,
                "time_visible_seconds": time_visible,
                "time_percentage": round((time_visible / duration) * 100, 2) if duration > 0 else 0,
                "total_detections": self.brand_detections[brand],
                "average_confidence": avg_conf,
            }

        # Find the brand with the most frames visible
        most_frequent = max(brands.items(), key=lambda x: x[1]["frames_visible"])[0] if brands else ""

        return {
            "video_info": {
                "fps": self.fps,
                "total_frames": self.total_frames,
                "duration_seconds": round(duration, 2),
            },
            "brands": brands,
            "most_frequent_brand": most_frequent,
            "most_frequent_count": brands[most_frequent]["frames_visible"] if most_frequent else 0,
            "total_brands_detected": len(brands),
        }

    def get_metrics_text(self) -> str:
        """
        Generate a human-readable text summary of all metrics.

        This text is used as input for the AI agent's analysis and
        is also printed to the console during processing.

        Returns:
            Formatted string with all brand metrics.
        """
        metrics = self.get_metrics()
        info = metrics["video_info"]

        lines = [
            "=" * 60,
            "  VIDEO ANALYSIS RESULTS",
            "=" * 60,
            f"  Duration: {info['duration_seconds']}s ({info['duration_seconds']/60:.1f} min)",
            f"  FPS: {info['fps']}",
            f"  Total Frames: {info['total_frames']}",
            f"  Brands Detected: {metrics['total_brands_detected']}",
            f"  Most Frequent: {metrics['most_frequent_brand']}",
            "=" * 60,
        ]

        for brand, data in sorted(metrics["brands"].items(),
                                    key=lambda x: x[1]["time_visible_seconds"],
                                    reverse=True):
            lines.append(f"\n[{brand}]")
            lines.append(f"  Frames Visible: {data['frames_visible']}")
            lines.append(f"  Time Visible: {data['time_visible_seconds']}s ({data['time_percentage']}%)")
            lines.append(f"  Total Detections: {data['total_detections']}")
            lines.append(f"  Avg Confidence: {data['average_confidence']}")

        return "\n".join(lines)
