import os
from collections import Counter
import pandas as pd

# Daftar nama kelas sesuai ID pada label YOLO11
class_names = [
    'dislocation',
    'grater trochanter',
    'intertrochanteric',
    'lesser trochanter',
    'neck',
    'normal',
    'subtrochanteric'
]

# Nama kelas target yang ingin dianalisis
target_class = 'subtrochanteric'

# Path folder yang berisi file-file label .txt
label_dir = r'Datasets/Raw/train/labels'

# Hitung kombinasi kelas per gambar
comb_counter = Counter()
for fname in os.listdir(label_dir):
    if not fname.lower().endswith('.txt'):
        continue
    with open(os.path.join(label_dir, fname), 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        combo = ('<no_object>',)
    else:
        ids = {int(line.split()[0]) for line in lines}
        combo = tuple(sorted(class_names[i] for i in ids))
    comb_counter[combo] += 1

# Buat DataFrame total distribusi kombinasi kelas
total_rows = [
    {'Kombinasi Kelas': ' + '.join(combo), 'Jumlah Gambar': count}
    for combo, count in comb_counter.items()
]
df_total = pd.DataFrame(total_rows)
df_total = df_total.sort_values(by='Jumlah Gambar', ascending=False).reset_index(drop=True)

# Filter hanya baris yang mengandung kelas target
df_target = df_total[df_total['Kombinasi Kelas'].str.contains(target_class)]

# Hitung total gambar yang mengandung kelas target (tanpa peduli kombinasi lain)
total_with_target = df_target['Jumlah Gambar'].sum()

# Tampilkan hasil
print(f"Distribusi kombinasi yang mengandung '{target_class}':")
print(df_target.to_markdown(index=False, tablefmt='github'))
print(f"\nTotal gambar yang mengandung '{target_class}': {total_with_target}")
