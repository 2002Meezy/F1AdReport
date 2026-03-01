"""
auto_label.py — Semi-automatic dataset labeling using a vision-language model.

Extracts frames from F1 videos and uses Qwen3 VL (via LM Studio) to identify
brand logos and generate YOLO-format annotations. This is useful for quickly
expanding the training dataset for under-represented brand classes.

Usage:
    python auto_label.py --source input/video.mp4
    python auto_label.py --source input/video.mp4 --max_frames 500 --frame_skip 25
"""

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

import cv2
import requests

# Classes YOLO — mesma ordem do data2.yaml
CLASS_NAMES = [
    "AWS", "American Express", "Aramco", "Cripto.com", "DHL",
    "ETIHAD", "Ferrari", "Globant", "Haas", "Heiniken",
    "KitKat", "LV", "Lenovo", "MSC cruises", "Mercedes",
    "Paramount+", "Pirelli", "Qatar Airways", "Rolex",
    "Salesforce", "TAG Heuer", "Vegas", "santander"
]

CLASS_NAME_TO_ID = {name.lower(): idx for idx, name in enumerate(CLASS_NAMES)}
# Add common variations
CLASS_NAME_TO_ID.update({
    "crypto.com": 3, "cripto": 3,
    "heineken": 9, "heiniken": 9,
    "louis vuitton": 11, "lv": 11,
    "msc": 13, "msc cruises": 13,
    "paramount": 15, "paramount plus": 15,
    "tag heuer": 20, "tagheuer": 20,
    "las vegas": 21, "vegas": 21,
    "american express": 1, "amex": 1,
    "qatar": 17, "qatar airways": 17,
    "kitkat": 10, "kit kat": 10,
    "etihad": 5, "etihad airways": 5,
})


def encode_image_base64(image) -> str:
    """Converte imagem OpenCV para base64."""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def query_vlm(image_b64: str, lm_studio_url: str, model: str) -> str:
    """
    Envia imagem para o Qwen3 VL via LM Studio e pede identificação de marcas.

    Retorna a resposta de texto do modelo.
    """
    prompt = f"""Analyze this Formula 1 racing image. Identify ALL brand logos/sponsors visible.

For EACH brand you see, provide:
- brand name
- approximate location as bounding box in format [x_center, y_center, width, height] where values are between 0.0 and 1.0 (relative to image size)

Known F1 brands to look for: {', '.join(CLASS_NAMES)}

Respond ONLY in this JSON format, no other text:
{{"detections": [{{"brand": "BrandName", "bbox": [x_center, y_center, width, height], "confidence": "high/medium/low"}}]}}

If no brands are visible, respond: {{"detections": []}}"""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    try:
        resp = requests.post(
            f"{lm_studio_url}/chat/completions",
            json=payload,
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content']
    except requests.exceptions.Timeout:
        print("  [!] Timeout -- model took too long", flush=True)
        return '{"detections": []}'
    except Exception as e:
        print(f"  [!] API error: {e}", flush=True)
        return '{"detections": []}'


def parse_response(response_text: str) -> list[dict]:
    """Extrai detecções da resposta do modelo."""
    # Try to find JSON in the response
    try:
        # Try direct parse
        data = json.loads(response_text)
        return data.get('detections', [])
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in text
    json_match = re.search(r'\{[\s\S]*"detections"[\s\S]*\}', response_text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return data.get('detections', [])
        except json.JSONDecodeError:
            pass

    return []


def detection_to_yolo(detection: dict, img_w: int, img_h: int) -> str | None:
    """
    Converte uma detecção para formato YOLO.

    Returns:
        String "class_id x_center y_center width height" ou None se inválido.
    """
    brand = detection.get('brand', '').strip()
    bbox = detection.get('bbox', [])

    if not brand or len(bbox) != 4:
        return None

    # Find class ID
    brand_lower = brand.lower()
    class_id = CLASS_NAME_TO_ID.get(brand_lower)

    if class_id is None:
        # Try partial match
        for name, idx in CLASS_NAME_TO_ID.items():
            if name in brand_lower or brand_lower in name:
                class_id = idx
                break

    if class_id is None:
        return None

    # Validate bbox values (should be 0-1 range)
    try:
        x_c, y_c, w, h = [float(v) for v in bbox]
    except (ValueError, TypeError):
        return None

    # Clamp to valid range
    x_c = max(0.0, min(1.0, x_c))
    y_c = max(0.0, min(1.0, y_c))
    w = max(0.01, min(1.0, w))
    h = max(0.01, min(1.0, h))

    return f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"


def process_video(source: Path, output_dir: Path, lm_studio_url: str,
                  model: str, frame_skip: int = 30, max_frames: int = 200,
                  start_frame: int = 0):
    """
    Processa vídeo e gera anotações YOLO usando Qwen3 VL.

    Args:
        source: Caminho do vídeo.
        output_dir: Diretório de saída.
        lm_studio_url: URL da API LM Studio.
        model: Identificador do modelo.
        frame_skip: Processar 1 a cada N frames.
        max_frames: Máximo de frames a processar.
        start_frame: Frame inicial.
    """
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"Error: Cannot open {source}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'='*60}")
    print(f"  Auto-Annotation with Qwen3 VL")
    print(f"{'='*60}")
    print(f"  Video:       {source.name}")
    print(f"  Resolution:  {width}x{height} @ {fps:.1f} FPS")
    print(f"  Total:       {total_frames} frames")
    print(f"  Processing:  1 every {frame_skip} frames")
    print(f"  Max frames:  {max_frames}")
    print(f"  Model:       {model}")
    print(f"  Output:      {output_dir}")
    print(f"{'='*60}\n")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    processed = 0
    total_detections = 0
    brand_counts = {}

    while processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # Encode frame
        image_b64 = encode_image_base64(frame)

        # Query VLM
        t0 = time.time()
        response = query_vlm(image_b64, lm_studio_url, model)
        elapsed = time.time() - t0

        # Parse detections
        detections = parse_response(response)

        # Convert to YOLO format
        yolo_lines = []
        for det in detections:
            line = detection_to_yolo(det, width, height)
            if line:
                yolo_lines.append(line)
                brand = det.get('brand', 'unknown')
                brand_counts[brand] = brand_counts.get(brand, 0) + 1

        total_detections += len(yolo_lines)

        # Save image and label
        frame_name = f"{source.stem}_frame{frame_idx:06d}"
        img_path = images_dir / f"{frame_name}.jpg"
        lbl_path = labels_dir / f"{frame_name}.txt"

        cv2.imwrite(str(img_path), frame)
        lbl_path.write_text('\n'.join(yolo_lines) if yolo_lines else '')

        status = f"[OK] {len(yolo_lines)} brands" if yolo_lines else "[--] empty"
        print(f"  [{processed+1}/{max_frames}] Frame {frame_idx} | "
              f"{status} | {elapsed:.1f}s", flush=True)

        processed += 1
        frame_idx += 1

    cap.release()

    # Summary
    print(f"\n{'='*60}")
    print(f"  Auto-Annotation Complete!")
    print(f"{'='*60}")
    print(f"  Frames processed: {processed}")
    print(f"  Total detections: {total_detections}")
    print(f"  Images saved to:  {images_dir}")
    print(f"  Labels saved to:  {labels_dir}")

    if brand_counts:
        print(f"\n  Brand counts:")
        for brand, count in sorted(brand_counts.items(), key=lambda x: -x[1]):
            print(f"    {brand}: {count}")

    print(f"\n  Next steps:")
    print(f"  1. Review labels in {labels_dir}")
    print(f"  2. Copy images + labels to datasets/f1_ads_v2/train/")
    print(f"  3. Retrain: python train_v2.py")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-annotate F1 video frames using Qwen3 VL")
    parser.add_argument('--source', type=str, required=True,
                        help='Path to video file')
    parser.add_argument('--output', type=str, default='datasets/auto_annotated',
                        help='Output directory')
    parser.add_argument('--lm_studio_url', type=str,
                        default='http://26.198.160.131:1234/v1',
                        help='LM Studio API URL')
    parser.add_argument('--model', type=str, default='qwen/qwen3-vl-4b',
                        help='Model identifier in LM Studio')
    parser.add_argument('--frame_skip', type=int, default=30,
                        help='Process 1 in every N frames (default: 30 = ~1/sec)')
    parser.add_argument('--max_frames', type=int, default=200,
                        help='Max frames to process')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Frame to start from')
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"Error: Video not found at {source}")
        sys.exit(1)

    # Test LM Studio connection
    print("Testing LM Studio connection...", flush=True)
    try:
        resp = requests.get(f"{args.lm_studio_url}/models", timeout=5)
        resp.raise_for_status()
        models = resp.json()
        print(f"Connected! Available models: "
              f"{[m['id'] for m in models.get('data', [])]}", flush=True)
    except Exception as e:
        print(f"Error: Cannot connect to LM Studio at {args.lm_studio_url}: {e}")
        print("Make sure LM Studio is running with the Qwen3 VL model loaded.")
        sys.exit(1)

    process_video(
        source=source,
        output_dir=Path(args.output),
        lm_studio_url=args.lm_studio_url,
        model=args.model,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
    )


if __name__ == '__main__':
    main()
