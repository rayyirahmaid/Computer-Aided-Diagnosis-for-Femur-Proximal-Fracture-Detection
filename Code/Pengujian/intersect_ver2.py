import os
from pathlib import Path

def yolo_to_bbox(label_line):
    """
    Convert YOLO label line to (x_min, y_min, x_max, y_max)
    """
    parts = label_line.strip().split()
    if len(parts) < 5:
        return None
    _, x_c, y_c, w, h = parts
    x_c, y_c, w, h = map(float, (x_c, y_c, w, h))
    return (x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2)

def area(bbox):
    """
    Compute area of a bounding box
    """
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

def intersect(a, b):
    """
    Compute intersection area of two bboxes
    """
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def process_pair(pred, gt):
    """
    Compute (tp, fp, fn) for one pred–GT pair using the 6-step rules
    """
    ia = intersect(pred, gt)
    a_pred, a_gt = area(pred), area(gt)
    # No intersection
    if ia == 0.0:
        return 0.0, a_pred, a_gt
    # pred inside gt
    if pred[0] >= gt[0] and pred[1] >= gt[1] and pred[2] <= gt[2] and pred[3] <= gt[3]:
        return a_pred, 0.0, a_gt - a_pred
    # gt inside pred
    if gt[0] >= pred[0] and gt[1] >= pred[1] and gt[2] <= pred[2] and gt[3] <= pred[3]:
        return a_gt, a_pred - a_gt, 0.0
    # partial overlap
    return ia, a_pred - ia, a_gt - ia

def read_boxes(path):
    """
    Read YOLO labels from file into list of bboxes
    """
    boxes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            bb = yolo_to_bbox(line)
            if bb:
                boxes.append(bb)
    return boxes

def pair_metrics(pred_boxes, gt_boxes):
    """
    For each pred box, match to best GT and produce per-pair metrics.
    Then add unmatched GT boxes as (0,0,area_gt).
    Returns list of (tp, fp, fn, precision, recall).
    """
    gt_matched = [False] * len(gt_boxes)
    results = []

    # Match each prediction to its best GT
    for pb in pred_boxes:
        best_tp = best_fp = best_fn = 0.0
        best_i = -1
        for i, gb in enumerate(gt_boxes):
            tp, fp, fn = process_pair(pb, gb)
            if tp > best_tp:
                best_tp, best_fp, best_fn = tp, fp, fn
                best_i = i
        # per-pair precision & recall
        prec = best_tp / (best_tp + best_fp) if (best_tp + best_fp) > 0 else 0.0
        rec  = best_tp / (best_tp + best_fn) if (best_tp + best_fn) > 0 else 0.0
        results.append((best_tp, best_fp, best_fn, prec, rec))
        if best_i >= 0 and not gt_matched[best_i]:
            gt_matched[best_i] = True

    # Unmatched GT boxes → produce one result each
    for matched, gb in zip(gt_matched, gt_boxes):
        if not matched:
            a_gt = area(gb)
            results.append((0.0, 0.0, a_gt, 0.0, 0.0))

    return results

def process_files(pred_file, gt_file):
    """
    Read boxes from two files and return per-pair metrics list.
    """
    pred = read_boxes(pred_file)
    gt   = read_boxes(gt_file)
    if not pred and not gt:
        return []
    return pair_metrics(pred, gt)

def main():
    # Paths to Prediction and GT folders
    pred_folder = "Pengujian/Hasil Anotasi Dokter/Detection Result/labels"
    gt_folder   = "Pengujian/Hasil Anotasi Dokter/User Training/labels"

    all_precisions = []
    all_recalls    = []

    for pf in Path(pred_folder).glob("*.txt"):
        gf = Path(gt_folder) / pf.name
        if not gf.exists():
            continue
        metrics = process_files(pf, gf)
        print(f"\nFile: {pf.name}")
        for idx, (tp, fp, fn, p, r) in enumerate(metrics, start=1):
            print(f" Pair {idx:02d}: TP={tp:.6f}, FP={fp:.6f}, FN={fn:.6f}, "
                  f"Precision={p*100:.2f}%, Recall={r*100:.2f}%")
            all_precisions.append(p)
            all_recalls.append(r)

    # Compute and print average Precision & Recall
    if all_precisions:
        avg_p = sum(all_precisions) / len(all_precisions)
        avg_r = sum(all_recalls)    / len(all_recalls)
        print()
        print("=== HASIL AKHIR ===")
        print(f"\nAverage Precision over all pairs: {avg_p*100:.2f}%")
        print(f"Average Recall    over all pairs: {avg_r*100:.2f}%")

if __name__ == "__main__":
    main()