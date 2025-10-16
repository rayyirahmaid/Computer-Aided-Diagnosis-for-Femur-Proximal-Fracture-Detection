from roboflow import Roboflow
import yaml
import os
from PIL import Image, ImageOps
import shutil

# 1. Custom post-processing functions
def apply_exif_orientation(image_path, output_path):
    """Apply EXIF orientation correction and save to output path"""
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        img.save(output_path)

def apply_custom_padding(image_path, output_path):
    """Apply black padding to 960x960 resolution and save to output path"""
    with Image.open(image_path) as img:
        new_img = Image.new("RGB", (960, 960), (0, 0, 0))
        new_img.paste(img, ((960 - img.width) // 2, (960 - img.height) // 2))
        new_img.save(output_path)

# 2. Conditional processing based on annotations
def process_dataset(data_yaml, output_base_path):
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)

    base_path = 'Datasets/Raw'

    for split in ['train', 'valid']:
        image_dir = os.path.join(base_path, split, 'images')
        label_dir = os.path.join(base_path, split, 'labels')

        # Buat direktori output jika belum ada
        output_image_dir = os.path.join(output_base_path, split, 'images')
        output_label_dir = os.path.join(output_base_path, split, 'labels')
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

        for img_file in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_file)
            label_filename = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_filename)

            # Path untuk simpan hasil
            output_img_path = os.path.join(output_image_dir, img_file)
            output_lbl_path = os.path.join(output_label_dir, label_filename)

            # Analyze annotations
            has_intertrochanteric = False
            has_trochanter = False

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        class_id = int(line.strip().split()[0])
                        if class_id == 2:
                            has_intertrochanteric = True
                        if class_id in [1, 3]:
                            has_trochanter = True

            if not has_intertrochanteric:
                apply_exif_orientation(img_path, output_img_path)
                if has_trochanter:
                    apply_custom_padding(output_img_path, output_img_path)  # Padding dilakukan setelah orientasi
            else:
              shutil.copy(img_path, output_img_path)  # Salin langsung tanpa modifikasi jika mengandung intertrochanteric

            # Salin label apa adanya
            if os.path.exists(label_path):
                shutil.copy(label_path, output_lbl_path)

# 3. Jalankan pemrosesan
output_base = 'Datasets/auto-orient'  # Ganti dengan direktori tujuan yang diinginkan
process_dataset('Datasets/Raw/data.yaml', output_base)