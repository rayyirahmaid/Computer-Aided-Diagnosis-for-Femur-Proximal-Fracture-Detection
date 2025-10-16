import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict

def process_augmentations(base_name: str, ext: str, img: np.ndarray, 
                         label_content: str, aug_types: List[str], output_dir: Path,
                         target_ids: List[int]):
    """Proses semua augmentasi dan simpan hasil ke folder terpisah"""
    (output_dir/'images').mkdir(parents=True, exist_ok=True)
    (output_dir/'labels').mkdir(parents=True, exist_ok=True)
    
    aug_functions = {
        'flip': lambda x: (cv2.flip(x, 1), '_flip'),       # Horizontal
        'flip_v': lambda x: (cv2.flip(x, 0), '_flip_v'),   # Vertical
    }
    
    for aug in aug_types:
        processed_img, suffix = aug_functions[aug](img)
        new_name = f"{base_name}{suffix}{ext}"
        
        # Proses label sesuai augmentasi
        flipped_lines = []
        for line in label_content.split('\n'):
            if not line.strip():
                continue
            parts = line.strip().split()
            class_id = int(parts[0])
            if class_id not in target_ids:
                continue  # Hapus object selain target
            
            x_center = float(parts[1])
            y_center = float(parts[2])
            
            if aug == 'flip':
                x_center = 1 - x_center
            elif aug == 'flip_v':
                y_center = 1 - y_center
            
            new_parts = parts.copy()
            new_parts[1] = f"{x_center:.6f}"
            new_parts[2] = f"{y_center:.6f}"
            flipped_lines.append(' '.join(new_parts))
        
        new_label_content = '\n'.join(flipped_lines)

        # Simpan gambar dan label yang hanya berisi target class
        if flipped_lines:  # Simpan hanya jika ada object target
            cv2.imwrite(str(output_dir/'images'/new_name), processed_img)
            (output_dir/'labels'/f"{base_name}{suffix}.txt").write_text(new_label_content)

def augment_dataset(train_dir: Path, yaml_path: Path, target_classes: List[str], output_dir: Path):
    """Augmentasi data hanya untuk gambar yang mengandung kelas target"""
    with yaml_path.open() as f:
        classes = yaml.safe_load(f)['names']
    
    target_ids = [i for i, name in enumerate(classes) if name in target_classes]
    if not target_ids:
        raise ValueError(f"Kelas target {target_classes} tidak ditemukan dalam YAML.")
    
    images_dir = train_dir/'images'
    labels_dir = train_dir/'labels'
    
    for img_path in images_dir.glob('*'):
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
            
        base_name = img_path.stem
        label_path = labels_dir/f"{base_name}.txt"
        
        if not label_path.exists():
            continue
        
        label_content = label_path.read_text()
        contains_target = False
        for line in label_content.splitlines():
            if line.strip():
                class_id = int(line.split()[0])
                if class_id in target_ids:
                    contains_target = True
                    break
        
        if not contains_target:
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Jalankan augmentasi dan hanya simpan label target
        process_augmentations(
            base_name, img_path.suffix, img, label_content,
            ['flip', 'flip_v'], output_dir, target_ids
        )

def analyze_dataset(labels_dir: Path, classes: List[str]) -> Dict[str, int]:
    """Analisis distribusi kelas dalam folder label"""
    class_count = {cls: 0 for cls in classes}
    
    for label_file in labels_dir.glob('*.txt'):
        for line in label_file.read_text().splitlines():
            if line:
                class_id = int(line.split()[0])
                class_count[classes[class_id]] += 1
                
    return class_count

if __name__ == '__main__':
    # Path konfigurasi
    base_dir = Path('C:/RayFile/KuliahBro/Semester8/TugasAkhir/Datasets + YOLO code/YOLO_V2DATASET/Proximal-Femur-Fracture-2')
    train_dir = base_dir/'train'
    yaml_path = base_dir/'data.yaml'
    
    # Output folder
    augmented_dir = base_dir/'augmented'
    augmented_dir.mkdir(parents=True, exist_ok=True)
    
    # Eksekusi augmentasi
    augment_dataset(
        train_dir=train_dir,
        yaml_path=yaml_path,
        target_classes=['grater trochanter', 'lesser trochanter'],
        output_dir=augmented_dir
    )
    
    # Analisis distribusi
    with yaml_path.open() as f:
        classes = yaml.safe_load(f)['names']
    
    print("Distribusi Asli:")
    orig_dist = analyze_dataset(train_dir/'labels', classes)
    for cls in classes:
        print(f"{cls}: {orig_dist[cls]}")
    
    print("\nDistribusi Augmentasi:")
    aug_dist = analyze_dataset(augmented_dir/'labels', classes)
    for cls in classes:
        print(f"{cls}: {aug_dist[cls]}")
