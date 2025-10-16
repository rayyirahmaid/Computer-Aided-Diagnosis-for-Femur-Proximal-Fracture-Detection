#!/usr/bin/env python3
# Image_Analyzer.py

import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from sklearn.mixture import GaussianMixture

# Tentukan path dataset di sini
TRAIN_FOLDER = 'Datasets/Raw/train'
VALID_FOLDER = 'Datasets/Raw/valid'
OUTPUT_EXCEL = 'Output/Validation/validation_report.xlsx'

def compute_brightness(img: np.ndarray, method: str = 'luma') -> float:
    """
    Menghitung rata-rata brightness (luminance) gambar.

    Args:
        img: input citra BGR atau grayscale, dtype uint8/float.
        method: 'luma' (default) atau 'mean'.
            - 'luma': weighted average sesuai persepsi mata (BT.601).
            - 'mean': rata-rata linear R+G+B / 3.

    Returns:
        float: nilai brightness rata-rata.
    """
    # Konversi ke float untuk akurasi
    arr = img.astype(np.float32)
    if arr.ndim == 3:
        b, g, r = cv2.split(arr)
        if method == 'luma':
            # BT.601 luma coefficients
            gray = 0.114 * b + 0.587 * g + 0.299 * r
        else:
            gray = (r + g + b) / 3.0
    else:
        gray = arr

    return float(np.mean(gray))


def compute_snr(img: np.ndarray) -> float:
    """
    Menghitung SNR tanpa mask dengan memisahkan noise
    melalui high-pass residual.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) \
           if img.ndim == 3 else img.astype(np.float32)
    # Sinyal: rata-rata luminance
    signal = np.mean(gray)
    # Estimasi noise: residual dari Gaussian blur
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=3, sigmaY=3)
    noise = np.std(gray - smooth)
    return float(signal / (noise + 1e-12))


def compute_cnr(img: np.ndarray, n_components: int = 2) -> float:
    """
    Menghitung CNR dengan otomatis memisahkan foreground (tulang)
    dan background lewat Gaussian Mixture Model (GMM).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) \
           if img.ndim == 3 else img.astype(np.float32)
    pixels = gray.flatten().reshape(-1, 1)
    # Fit GMM dua komponen: satu untuk bone, satu untuk background
    gmm = GaussianMixture(n_components=n_components, covariance_type='tied', random_state=0)
    labels = gmm.fit_predict(pixels)
    comp_means = np.sort(gmm.means_.ravel())
    mean_bg, mean_fg = comp_means[0], comp_means[-1]
    # Noise: std dev residual seperti di SNR
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=3, sigmaY=3)
    noise = np.std(gray - smooth)
    return float(abs(mean_fg - mean_bg) / (noise + 1e-12))

def is_grayscale(img: np.ndarray) -> bool:
    """Cek apakah gambar grayscale."""
    if img.ndim == 2:
        return True
    b, g, r = cv2.split(img)
    return np.array_equal(b, g) and np.array_equal(g, r)

def parse_yolo_label(path: str) -> bool:
    """Validasi format YOLO: 'class x_center y_center width height' dalam rentang [0,1]."""
    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    return False
                *coords, = map(float, parts[1:])
                if any(v < 0 or v > 1 for v in coords):
                    return False
        return True
    except:
        return False

def analyze_folder(folder: str) -> dict:
    img_paths = glob(os.path.join(folder, 'images', '*.jpg')) + \
                glob(os.path.join(folder, 'images', '*.png'))
    lbl_paths = glob(os.path.join(folder, 'labels', '*.txt'))

    size_counts = {}
    orient_counts = {'landscape': 0, 'portrait': 0}
    brightness_bins = {'0-50': 0, '50-100': 0, '100-150': 0, '150-200': 0, '200-255': 0}
    snr_bins = {'0-5': 0, '5-10': 0, '10-20': 0, '20-50': 0, '50+': 0}
    cnr_bins = {'0-5': 0, '5-10': 0, '10-20': 0, '20-50': 0, '50+': 0}
    color_counts = {'rgb': 0, 'grayscale': 0}
    img_has_label = 0
    label_has_img = 0
    yolo_format_ok = 0

    lbl_map = {os.path.splitext(os.path.basename(p))[0]: p for p in lbl_paths}
    img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in img_paths}

    for img_path in img_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]

        # Ukuran
        key_size = f"{w}x{h}"
        size_counts[key_size] = size_counts.get(key_size, 0) + 1

        # Orientasi
        orient_counts['landscape' if w >= h else 'portrait'] += 1

        # Brightness
        b = compute_brightness(img)
        for k in brightness_bins:
            if '-' in k:
                low, high = map(float, k.split('-'))
            else:
                low, high = float(k.replace('+','')), np.inf
            if b >= low and b < high:
                brightness_bins[k] += 1
                break

        # SNR
        snr = compute_snr(img)
        for k in snr_bins:
            if '+' in k:
                if snr >= float(k.replace('+','')):
                    snr_bins[k] += 1
                    break
            else:
                low, high = map(float, k.split('-'))
                if snr >= low and snr < high:
                    snr_bins[k] += 1
                    break

        # CNR
        cnr = compute_cnr(img)
        for k in cnr_bins:
            if '+' in k:
                if cnr >= float(k.replace('+','')):
                    cnr_bins[k] += 1
                    break
            else:
                low, high = map(float, k.split('-'))
                if cnr >= low and cnr < high:
                    cnr_bins[k] += 1
                    break

        # RGB vs Grayscale
        color_counts['grayscale' if is_grayscale(img) else 'rgb'] += 1

        # Label ada?
        if base in lbl_map:
            img_has_label += 1

    for base, lbl_path in lbl_map.items():
        if base in img_map:
            label_has_img += 1
        if parse_yolo_label(lbl_path):
            yolo_format_ok += 1

    return {
        'size_counts': size_counts,
        'orientation': orient_counts,
        'brightness': brightness_bins,
        'snr': snr_bins,
        'cnr': cnr_bins,
        'color_format': color_counts,
        'img_with_label': img_has_label,
        'label_with_img': label_has_img,
        'yolo_valid_count': yolo_format_ok,
        'total_images': len(img_paths),
        'total_labels': len(lbl_paths)
    }

def to_excel(results: dict, output_path: str):
    # Buat satu sheet per jenis validasi
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # iterate kategori
        for category in results['train'].keys():
            train_data = results['train'][category]
            valid_data = results['valid'][category]
            # Jika dict, buat DataFrame dengan kolom key, train, valid
            if isinstance(train_data, dict):
                keys = sorted(set(train_data.keys()) | set(valid_data.keys()))
                rows = []
                for k in keys:
                    rows.append({
                        category: k,
                        'train': train_data.get(k, 0),
                        'valid': valid_data.get(k, 0)
                    })
                df = pd.DataFrame(rows)
            else:
                # scalar kasus total_images, total_labels, img_with_label, label_with_img, yolo_valid_count
                df = pd.DataFrame({
                    category: ['value'],
                    'train': [train_data],
                    'valid': [valid_data]
                })
            sheet = category[:31]
            df.to_excel(writer, sheet_name=sheet, index=False)

if __name__ == '__main__':
    datasets = {
        'train': TRAIN_FOLDER,
        'valid': VALID_FOLDER
    }
    all_results = {name: analyze_folder(path) for name, path in datasets.items()}
    to_excel(all_results, OUTPUT_EXCEL)
    print(f"Hasil validasi disimpan di {OUTPUT_EXCEL}")