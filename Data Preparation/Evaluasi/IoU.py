import os

def yolo_to_bbox(line, img_width, img_height):
    """
    Konversi satu baris format YOLO (class x_center y_center w h)
    ke bounding box pixel (x_min, y_min, x_max, y_max).
    """
    parts = line.strip().split()
    _, x_ctr, y_ctr, w, h = map(float, parts)
    x_ctr *= img_width
    y_ctr *= img_height
    w *= img_width
    h *= img_height
    x_min = x_ctr - w / 2
    y_min = y_ctr - h / 2
    x_max = x_ctr + w / 2
    y_max = y_ctr + h / 2
    return (x_min, y_min, x_max, y_max)

def compute_iou(boxA, boxB):
    """
    Hitung IoU antara dua bbox pixel: (x_min, y_min, x_max, y_max).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter_area
    return inter_area / union if union > 0 else 0

def read_yolo_file(path):
    """Baca file YOLO, kembalikan daftar baris non-kosong."""
    with open(path, 'r') as f:
        return [l for l in f.read().splitlines() if l.strip()]

def match_best_pairs(bboxes1, bboxes2, top_k):
    """
    Pilih top_k pasangan (i,j) bboxes1 vs bboxes2 berdasar IoU tertinggi,
    tanpa mengulang i atau j.
    """
    pairs = []
    for i, b1 in enumerate(bboxes1):
        for j, b2 in enumerate(bboxes2):
            iou = compute_iou(b1, b2)
            pairs.append((iou, i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)

    used1, used2 = set(), set()
    selected = []
    for iou, i, j in pairs:
        if len(selected) >= top_k:
            break
        if i not in used1 and j not in used2:
            used1.add(i); used2.add(j)
            selected.append((i, j, iou))
    return selected

def evaluate(label_path1, label_path2, img_width, img_height, iou_threshold=0.5):
    # Baca dan konversi ke bbox pixel
    lines1 = read_yolo_file(label_path1)
    lines2 = read_yolo_file(label_path2)
    bboxes1 = [yolo_to_bbox(l, img_width, img_height) for l in lines1]
    bboxes2 = [yolo_to_bbox(l, img_width, img_height) for l in lines2]

    # Pencocokan sebanyak jumlah ground truth (file2)
    top_k = len(bboxes2)
    matches = match_best_pairs(bboxes1, bboxes2, top_k)

    # Hitung TP, FP, FN
    tp = sum(1 for (_, _, iou) in matches if iou >= iou_threshold)
    fn = len(bboxes2) - tp
    fp = len(bboxes1) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Output detail
    print(f"Threshold IoU untuk TP: {iou_threshold}")
    print(f"Total Prediksi (File1): {len(bboxes1)}")
    print(f"Total Ground Truth (File2): {len(bboxes2)}")
    print(f"True Positives  (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precision (P): {precision:.4f}")
    print(f"Recall    (R): {recall:.4f}\n")

    print("Detail pasangan terpilih:")
    for idx, (i, j, iou) in enumerate(matches, 1):
        status = "TP" if iou >= iou_threshold else "FP/FN"
        print(f"  Pair {idx}: Prediksi {i+1} ↔ GT {j+1}  IoU={iou:.4f}  [{status}]")

if __name__ == "__main__":
    # Sesuaikan dimensi gambar dan jalur file di bawah ini:
    img_w, img_h = 1920, 1080
    label_file_1 = "femur_fracture_app/Hasil Deteksi/Detection Mode/labels/N (51).txt"
    label_file_2 = "Pengujian/Hasil Anotasi Dokter/User Training/labels/N (51).txt"

    if not os.path.exists(label_file_1) or not os.path.exists(label_file_2):
        print("Pastikan kedua file label ada di path yang benar.")
    else:
        evaluate(label_file_1, label_file_2, img_w, img_h, iou_threshold=0.5)
