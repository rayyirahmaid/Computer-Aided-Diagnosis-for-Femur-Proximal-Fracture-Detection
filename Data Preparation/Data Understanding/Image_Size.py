import os
import math
from PIL import Image, ImageOps
import pandas as pd

# Folder tempat gambar berada
folder_path = 'Datasets/Raw/valid/images'

# Daftar ukuran resolusi umum dengan rentang luas
COMMON_SIZES = {
    'SD (640×480)': (640, 480),
    'HD (1280×720)': (1280, 720),
    'Full HD (1920×1080)': (1920, 1080),
    '2K (2560×1440)': (2560, 1440),
    '4K (3840×2160)': (3840, 2160)
}

def get_nearest_size_label(width, height):
    """Pilih ukuran umum yang paling mendekati dimensi width×height."""
    best_label = None
    smallest_diff = float('inf')
    for label, (cw, ch) in COMMON_SIZES.items():
        # gunakan selisih relatif untuk mempertimbangkan skala
        diff = abs(width - cw)/cw + abs(height - ch)/ch
        if diff < smallest_diff:
            smallest_diff = diff
            best_label = label
    return best_label

# Struktur data
records = []
size_count = {}
generalized_count = {}

for fname in os.listdir(folder_path):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    path = os.path.join(folder_path, fname)
    try:
        file_size = os.path.getsize(path)
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            width, height = img.size

        # Tentukan label ukuran paling mendekati
        size_label = get_nearest_size_label(width, height)
        # Hitung megapixel dengan satu desimal
        megapixels = round((width * height) / 1_000_000, 1)

        records.append([
            fname,
            file_size,
            width,
            height,
            size_label,
            megapixels
        ])

        # Frekuensi ukuran aktual
        key_actual = f'{width}×{height}'
        size_count[key_actual] = size_count.get(key_actual, 0) + 1

        # Frekuensi ukuran umum terdekat
        generalized_count[size_label] = generalized_count.get(size_label, 0) + 1

    except Exception as e:
        print(f'Error processing {fname}: {e}')

# Buat DataFrame
df = pd.DataFrame(records, columns=[
    'Filename',
    'File Size (bytes)',
    'Width',
    'Height',
    'Generalized Size',
    'Megapixels'
])
df_actual = pd.DataFrame(size_count.items(), columns=['Actual Size (WxH)', 'Count'])
df_generalized = pd.DataFrame(generalized_count.items(), columns=['Generalized Size', 'Count'])

# Simpan ke Excel
os.makedirs('Output/Validation', exist_ok=True)
output_excel = 'Output/Validation/Image_Size_Generalized.xlsx'
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Image Details', index=False)
    df_actual.to_excel(writer, sheet_name='Actual Size Count', index=False)
    df_generalized.to_excel(writer, sheet_name='Generalized Size Count', index=False)

print(f'Data berhasil disimpan ke {output_excel}')
print(f'Current directory: {os.getcwd()}')