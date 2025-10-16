import os
import yaml
from collections import Counter

def parse_yaml_classes(yaml_path):
    """Membaca file YAML dan ekstrak daftar kelas"""
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    
    if 'names' not in data:
        raise ValueError("File YAML tidak mengandung key 'names'")
    
    return data['names']

def count_objects_in_labels(labels_dir, class_names):
    """Menghitung jumlah objek per kelas dari file label"""
    counter = Counter()
    
    for filename in os.listdir(labels_dir):
        if not filename.endswith('.txt'):
            continue
            
        filepath = os.path.join(labels_dir, filename)
        
        try:
            with open(filepath, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= len(class_names):
                        print(f"Peringatan: Class ID {class_id} di file {filename} di luar jangkauan")
                        continue
                        
                    counter[class_id] += 1
        except Exception as e:
            print(f"Gagal memproses file {filename}: {str(e)}")
    
    return counter

def main():
    # Ganti path berikut sesuai lokasi file dan folder Anda
    yaml_path = 'Datasets/Raw/data.yaml'  # Path ke file YAML
    labels_dir = 'Datasets/Raw/valid/labels'    # Path ke folder label
    
    try:
        # Parse kelas dari YAML
        class_names = parse_yaml_classes(yaml_path)
        print(f"Ditemukan {len(class_names)} kelas:")
        for i, name in enumerate(class_names):
            print(f"{i}: {name}")
            
        # Hitung objek
        counter = count_objects_in_labels(labels_dir, class_names)
        
        # Tampilkan hasil
        print("\nJumlah objek per kelas:")
        total = 0
        for class_id, count in counter.most_common():
            print(f"{class_names[class_id]}: {count}")
            total += count
        print(f"\nTotal objek: {total}")
        
    except FileNotFoundError:
        print("Error: File atau folder tidak ditemukan")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
