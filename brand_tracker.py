"""
BrandTracker — Coleta métricas de detecção de marcas frame a frame.

Cada frame processado pelo YOLO é passado ao BrandTracker, que registra:
- Marca detectada (nome da classe)
- Número do frame
- Confiança da detecção

Ao final, calcula:
- Tempo total de exibição por marca
- Frequência de aparições
- Marca mais frequente
"""

from collections import defaultdict


class BrandTracker:
    """Rastreia detecções de marcas ao longo dos frames de um vídeo."""

    def __init__(self, class_names: list[str], fps: float = 30.0):
        """
        Args:
            class_names: Lista de nomes das classes (marcas) na ordem do modelo YOLO.
            fps: Frames por segundo do vídeo fonte.
        """
        self.class_names = class_names
        self.fps = fps

        # frame_number -> set of brand names detected in that frame
        self.frame_detections: dict[int, set[str]] = defaultdict(set)
        # brand_name -> list of confidence scores (one per detection)
        self.brand_confidences: dict[str, list[float]] = defaultdict(list)
        # Total frames processed
        self.total_frames = 0

    def update(self, frame_number: int, results) -> None:
        """
        Registra as detecções de um frame.

        Args:
            frame_number: Índice do frame atual.
            results: Objeto de resultado do YOLO para um único frame.
        """
        self.total_frames = max(self.total_frames, frame_number + 1)

        if results.boxes is None or len(results.boxes) == 0:
            return

        boxes = results.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())

            if cls_id < len(self.class_names):
                brand_name = self.class_names[cls_id]
            else:
                brand_name = f"class_{cls_id}"

            self.frame_detections[frame_number].add(brand_name)
            self.brand_confidences[brand_name].append(conf)

    def get_metrics(self) -> dict:
        """
        Calcula e retorna todas as métricas de tracking.

        Returns:
            Dicionário com métricas por marca e métricas globais.
        """
        video_duration = self.total_frames / self.fps if self.fps > 0 else 0

        # Contagem de frames em que cada marca aparece
        brand_frame_count: dict[str, int] = defaultdict(int)
        for frame_brands in self.frame_detections.values():
            for brand in frame_brands:
                brand_frame_count[brand] += 1

        # Total de detecções por marca (pode ter múltiplas por frame)
        brand_detection_count: dict[str, int] = {
            brand: len(confs) for brand, confs in self.brand_confidences.items()
        }

        # Métricas por marca
        brands = {}
        for brand in sorted(brand_detection_count.keys()):
            frames_visible = brand_frame_count[brand]
            time_visible = frames_visible / self.fps if self.fps > 0 else 0
            total_detections = brand_detection_count[brand]
            confs = self.brand_confidences[brand]
            avg_confidence = sum(confs) / len(confs) if confs else 0
            time_percentage = (time_visible / video_duration * 100) if video_duration > 0 else 0

            brands[brand] = {
                "frames_visible": frames_visible,
                "time_visible_seconds": round(time_visible, 2),
                "total_detections": total_detections,
                "average_confidence": round(avg_confidence, 4),
                "time_percentage": round(time_percentage, 2),
            }

        # Marca mais frequente (por frames visíveis)
        most_frequent = max(brand_frame_count, key=brand_frame_count.get) if brand_frame_count else None

        return {
            "video_info": {
                "total_frames": self.total_frames,
                "fps": self.fps,
                "duration_seconds": round(video_duration, 2),
            },
            "brands": brands,
            "most_frequent_brand": most_frequent,
            "most_frequent_count": brand_frame_count.get(most_frequent, 0) if most_frequent else 0,
            "total_brands_detected": len(brands),
        }

    def get_metrics_text(self) -> str:
        """Retorna as métricas formatadas como texto para uso pelo agente IA."""
        metrics = self.get_metrics()
        lines = []
        lines.append(f"=== Video Analysis Results ===")
        lines.append(f"Total Frames: {metrics['video_info']['total_frames']}")
        lines.append(f"FPS: {metrics['video_info']['fps']}")
        lines.append(f"Duration: {metrics['video_info']['duration_seconds']}s")
        lines.append(f"Total Brands Detected: {metrics['total_brands_detected']}")
        lines.append(f"Most Frequent Brand: {metrics['most_frequent_brand']} "
                      f"(visible in {metrics['most_frequent_count']} frames)")
        lines.append("")
        lines.append("=== Per-Brand Breakdown ===")

        for brand, data in metrics["brands"].items():
            lines.append(f"\n[{brand}]")
            lines.append(f"  Frames Visible: {data['frames_visible']}")
            lines.append(f"  Time Visible: {data['time_visible_seconds']}s ({data['time_percentage']}%)")
            lines.append(f"  Total Detections: {data['total_detections']}")
            lines.append(f"  Avg Confidence: {data['average_confidence']}")

        return "\n".join(lines)
