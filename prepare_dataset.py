import argparse
import os
import shutil
import zipfile
import json
from pathlib import Path
import random

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def clean_dir(path: Path):
    if path.exists():
        try:
            shutil.rmtree(path)
        except PermissionError:
            print(f"Warning: Could not fully clean {path}")
    path.mkdir(parents=True, exist_ok=True)

def find_source_dir(zip_path: Path, src_dir: Path | None, prefer_zip: bool = False) -> Path:
    if prefer_zip and zip_path and zip_path.exists():
        extract_dir = zip_path.with_suffix("")
        if extract_dir.exists():
            try:
                shutil.rmtree(extract_dir)
            except PermissionError:
                print(f"Warning: Could not clean extraction dir {extract_dir}")
        
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Extracting {zip_path} to {extract_dir}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
        except Exception as e:
            print(f"Error extracting zip: {e}")
            raise
            
        return extract_dir
    
    if src_dir and src_dir.exists():
        return src_dir
        
    if zip_path and zip_path.exists():
        extract_dir = zip_path.with_suffix("")
        if not extract_dir.exists():
            print(f"Extracting {zip_path} to {extract_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
        return extract_dir
        
    raise FileNotFoundError("No valid source found. Provide --zip_path or --src_dir")

def convert_coco_to_yolo(box, img_w, img_h):
    # COCO: [x_min, y_min, width, height]
    # YOLO: [x_center, y_center, width, height] normalized
    x_min, y_min, w, h = box
    
    x_center = x_min + w / 2.0
    y_center = y_min + h / 2.0
    
    x = x_center / img_w
    y = y_center / img_h
    w = w / img_w
    h = h / img_h
    
    return x, y, w, h

def process_notes_json(source_dir: Path, json_path: Path) -> bool:
    print(f"Processing notes.json at {json_path}...")
    try:
        data = json.loads(json_path.read_text(encoding='utf-8'))
        
        if 'annotations' not in data or 'images' not in data:
            print("notes.json does not contain 'annotations' or 'images' keys. Skipping JSON conversion.")
            return False
            
        print(f"Found {len(data['images'])} images and {len(data['annotations'])} annotations in JSON.")
        
        # Map image id to filename and size
        img_map = {img['id']: img for img in data['images']}
        
        # Group annotations by image id
        ann_map = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in ann_map:
                ann_map[img_id] = []
            ann_map[img_id].append(ann)
            
        # Overwrite/Create .txt files
        count_created = 0
        for img_id, anns in ann_map.items():
            if img_id not in img_map:
                continue
                
            img_info = img_map[img_id]
            file_name = img_info['file_name']
            
            # Find the actual file in source_dir
            found_img = list(source_dir.rglob(file_name))
            if not found_img:
                # Try matching just the name without path
                name_only = Path(file_name).name
                found_img = list(source_dir.rglob(name_only))
                
            if not found_img:
                continue
                
            # Use the first match
            img_path = found_img[0]
            txt_path = img_path.with_suffix('.txt')
            
            img_w = img_info['width']
            img_h = img_info['height']
            
            lines = []
            for ann in anns:
                cat_id = ann['category_id']
                bbox = ann['bbox'] # [x, y, w, h]
                
                # Convert to YOLO
                x, y, w, h = convert_coco_to_yolo(bbox, img_w, img_h)
                
                # Clip to [0, 1]
                x = max(0, min(1, x))
                y = max(0, min(1, y))
                w = max(0, min(1, w))
                h = max(0, min(1, h))
                
                lines.append(f"{cat_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                
            txt_path.write_text("\n".join(lines), encoding='utf-8')
            count_created += 1
            
        print(f"Successfully generated {count_created} label files from notes.json")
        return True
        
    except Exception as e:
        print(f"Error processing notes.json: {e}")
        return False

def collect_pairs(root: Path) -> tuple[list[Path], list[Path]]:
    # 1. Try to process notes.json first to generate correct labels
    json_files = list(root.rglob('notes.json'))
    if json_files:
        for jf in json_files:
            if process_notes_json(root, jf):
                print("Using labels generated from notes.json")
                break

    image_map = {}
    label_map = {}
    
    print(f"Scanning for files in {root}...")
    
    for p in root.rglob('*'):
        if p.is_file():
            if p.suffix.lower() in IMG_EXTS:
                if p.stem not in image_map:
                    image_map[p.stem] = p
            elif p.suffix.lower() == '.txt' and p.name != 'classes.txt':
                label_map[p.stem] = p
                
    pairs = []
    print(f"Found {len(image_map)} images and {len(label_map)} label files.")

    # Analyze class distribution
    class_counts = {}
    for lbl_path in label_map.values():
        try:
            content = lbl_path.read_text(encoding='utf-8', errors='ignore')
            for line in content.splitlines():
                if line.strip():
                    parts = line.split()
                    if parts:
                        try:
                            cid = int(float(parts[0]))
                            class_counts[cid] = class_counts.get(cid, 0) + 1
                        except ValueError:
                            pass
        except Exception:
            pass
            
    print("\n=== Source Label Distribution ===")
    if not class_counts:
        print("No valid labels found in .txt files.")
    else:
        for cid in sorted(class_counts.keys()):
            print(f"  Class ID {cid}: {class_counts[cid]} occurrences")
    print("=================================\n")

    for stem, img_path in image_map.items():
        if stem in label_map:
            pairs.append((img_path, label_map[stem]))
        else:
            pairs.append((img_path, None))
            
    pairs.sort(key=lambda x: x[0].stem)
    
    images = [p[0] for p in pairs]
    labels = [p[1] for p in pairs]
    
    return images, labels

def read_class_names(root: Path) -> list[str]:
    # 1. Try notes.json
    for notes_file in root.rglob('notes.json'):
        if notes_file.is_file():
            try:
                data = json.loads(notes_file.read_text(encoding='utf-8'))
                categories = data.get('categories', [])
                if categories:
                    categories.sort(key=lambda x: x.get('id', 0))
                    max_id = max(c.get('id', 0) for c in categories)
                    names = [f"class{i}" for i in range(max_id + 1)]
                    for c in categories:
                        cid = c.get('id')
                        if cid is not None:
                            names[cid] = c.get('name', f"class{cid}")
                    
                    print(f"Loaded {len(names)} classes from notes.json: {names}")
                    return names
            except Exception:
                pass

    # 2. Try classes.txt
    for candidate in root.rglob('classes.txt'):
        if candidate.is_file():
            content = [ln.strip() for ln in candidate.read_text(encoding='utf-8', errors='ignore').splitlines() if ln.strip()]
            if content:
                print(f"Loaded {len(content)} classes from classes.txt")
                return content
            
    print("classes.txt/notes.json not found. Inferring classes from label files...")
    return ['object']

def write_yaml(train_images: Path, val_images: Path, names: list[str], out_path: Path):
    train_p = train_images.as_posix()
    val_p = val_images.as_posix()
    lines = []
    lines.append(f"train: {train_p}")
    lines.append(f"val: {val_p}")
    
    lines.append("names:")
    for n in names:
        lines.append(f"  - '{n}'")
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding='utf-8')

def split_and_copy(images: list[Path], labels: list[Path | None], out_root: Path, val_ratio: float, seed: int):
    try:
        clean_dir(out_root)
    except PermissionError:
        print(f"Warning: Could not clean {out_root}. Proceeding...")

    pairs = list(zip(images, labels))
    random.Random(seed).shuffle(pairs)
    
    n_total = len(pairs)
    n_val = max(1, int(n_total * val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    
    paths = {
        'train_images': out_root / 'train' / 'images',
        'train_labels': out_root / 'train' / 'labels',
        'val_images': out_root / 'val' / 'images',
        'val_labels': out_root / 'val' / 'labels',
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
        
    print(f"Copying {len(train_pairs)} training pairs...")
    for img, lbl in train_pairs:
        shutil.copy2(img, paths['train_images'] / img.name)
        if lbl:
            shutil.copy2(lbl, paths['train_labels'] / (img.stem + '.txt'))
        else:
            (paths['train_labels'] / (img.stem + '.txt')).touch()
        
    print(f"Copying {len(val_pairs)} validation pairs...")
    for img, lbl in val_pairs:
        shutil.copy2(img, paths['val_images'] / img.name)
        if lbl:
            shutil.copy2(lbl, paths['val_labels'] / (img.stem + '.txt'))
        else:
            (paths['val_labels'] / (img.stem + '.txt')).touch()
        
    return paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip_path', type=str, default=None)
    parser.add_argument('--src_dir', type=str, default=None, help="Path to already extracted data directory")
    parser.add_argument('--out_root', type=str, default=str(Path.cwd() / 'datasets' / 'f1_ads'))
    parser.add_argument('--config_out', type=str, default=str(Path.cwd() / 'configs' / 'data.yaml'))
    parser.add_argument('--val_ratio', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    zip_path = Path(args.zip_path) if args.zip_path else None
    src_dir = Path(args.src_dir) if args.src_dir else None
    out_root = Path(args.out_root)
    cfg_out = Path(args.config_out)

    if src_dir and src_dir.exists():
        print(f"Using existing source directory: {src_dir}")
        source = src_dir
    elif zip_path:
        source = find_source_dir(zip_path, None, prefer_zip=True)
    else:
        # Fallback default
        default_zip = Path(r"C:\Users\luizr\CascadeProjects\Visual\data.zip")
        if default_zip.exists():
             source = find_source_dir(default_zip, None, prefer_zip=True)
        else:
             raise ValueError("Must provide --zip_path or --src_dir")
    
    images, labels = collect_pairs(source)
    if not images:
        raise RuntimeError('No images found in the source.')
        
    names = read_class_names(source)
    print(f"Detected classes: {names}")
    
    paths = split_and_copy(images, labels, out_root, args.val_ratio, args.seed)
    
    write_yaml(Path(paths['train_images']), Path(paths['val_images']), names, cfg_out)
    
    print('Dataset preparation complete.')
    print(f"YAML config: {cfg_out}")

if __name__ == '__main__':
    main()
