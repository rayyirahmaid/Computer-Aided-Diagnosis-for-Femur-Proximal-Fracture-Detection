import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path

class YOLOConfusionMatrix:
    def __init__(self, pred_folder, gt_folder, class_names=None, iou_threshold=0.5):
        self.pred_folder = Path(pred_folder)
        self.gt_folder = Path(gt_folder)
        self.iou_threshold = iou_threshold
        self.class_names = class_names
        self.confusion_matrix = None
        self.classes = set()
    
    def parse_yolo_txt(self, txt_file_path):
        detections = []
        if not os.path.exists(txt_file_path):
            return detections
        with open(txt_file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    detections.append((class_id, x_center, y_center, width, height))
                    self.classes.add(class_id)
        return detections

    def yolo_to_bbox(self, x_center, y_center, width, height, img_width=1, img_height=1):
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        return x_min, y_min, x_max, y_max

    def calculate_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        intersection_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersection_area
        if union_area <= 0:
            return 0.0
        return intersection_area / union_area

    def match_detections(self, pred_detections, gt_detections):
        matched_pairs = []
        unmatched_preds = list(range(len(pred_detections)))
        unmatched_gts = list(range(len(gt_detections)))
        iou_matrix = np.zeros((len(pred_detections), len(gt_detections)))
        for i, pred in enumerate(pred_detections):
            pred_bbox = self.yolo_to_bbox(pred[1], pred[2], pred[3], pred[4])
            for j, gt in enumerate(gt_detections):
                gt_bbox = self.yolo_to_bbox(gt[1], gt[2], gt[3], gt[4])
                iou_matrix[i, j] = self.calculate_iou(pred_bbox, gt_bbox)
        while True:
            if len(unmatched_preds) == 0 or len(unmatched_gts) == 0:
                break
            best_iou = 0
            best_pred_idx = -1
            best_gt_idx = -1
            for i in unmatched_preds:
                for j in unmatched_gts:
                    if iou_matrix[i, j] > best_iou and iou_matrix[i, j] >= self.iou_threshold:
                        best_iou = iou_matrix[i, j]
                        best_pred_idx = i
                        best_gt_idx = j
            if best_pred_idx == -1:
                break
            matched_pairs.append((best_pred_idx, best_gt_idx))
            unmatched_preds.remove(best_pred_idx)
            unmatched_gts.remove(best_gt_idx)
        return matched_pairs, unmatched_preds, unmatched_gts

    def process_files(self):
        pred_files = list(self.pred_folder.glob("*.txt"))
        confusion_data = defaultdict(lambda: defaultdict(int))
        for pred_file in pred_files:
            gt_file = self.gt_folder / pred_file.name
            pred_detections = self.parse_yolo_txt(pred_file)
            gt_detections = self.parse_yolo_txt(gt_file)
            matched_pairs, unmatched_preds, unmatched_gts = self.match_detections(pred_detections, gt_detections)
            for pred_idx, gt_idx in matched_pairs:
                pred_class = pred_detections[pred_idx][0]
                gt_class = gt_detections[gt_idx][0]
                confusion_data[gt_class][pred_class] += 1
            for pred_idx in unmatched_preds:
                pred_class = pred_detections[pred_idx][0]
                confusion_data['background'][pred_class] += 1
            for gt_idx in unmatched_gts:
                gt_class = gt_detections[gt_idx][0]
                confusion_data[gt_class]['background'] += 1
        all_classes = sorted(list(self.classes)) + ['background']
        matrix_size = len(all_classes)
        self.confusion_matrix = np.zeros((matrix_size, matrix_size))
        class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
        for true_class, pred_dict in confusion_data.items():
            true_idx = class_to_idx[true_class]
            for pred_class, count in pred_dict.items():
                pred_idx = class_to_idx[pred_class]
                self.confusion_matrix[true_idx, pred_idx] = count
        self.class_indices = all_classes

    def plot_confusion_matrix(self, normalize=False, save_path=None, figsize=(10, 8)):
        if self.confusion_matrix is None:
            raise ValueError("Confusion matrix not computed. Run process_files() first.")
        matrix = self.confusion_matrix.copy()
        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums!=0)
        if self.class_names and len(self.class_names) >= len(self.classes):
            labels = [self.class_names[i] for i in sorted(self.classes)] + ['background']
        else:
            labels = [f'Class_{i}' for i in sorted(self.classes)] + ['background']
        plt.figure(figsize=figsize)
        sns.heatmap(matrix, 
                   annot=True, 
                   fmt='.2f' if normalize else '.0f',
                   cmap='Blues',
                   xticklabels=labels,
                   yticklabels=labels,
                   cbar_kws={'label': 'Normalized Count' if normalize else 'Count'})
        plt.title(f'Confusion Matrix (IoU threshold: {self.iou_threshold})')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def calculate_metrics(self):
        if self.confusion_matrix is None:
            raise ValueError("Confusion matrix not computed. Run process_files() first.")
        metrics = {}
        n_classes = len(self.class_indices) - 1  # Exclude background
        for i in range(n_classes):
            class_name = self.class_indices[i]
            tp = self.confusion_matrix[i, i]
            fp = np.sum(self.confusion_matrix[:, i]) - tp
            fn = np.sum(self.confusion_matrix[i, :]) - tp
            tn = np.sum(self.confusion_matrix) - tp - fp - fn
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            }
        total_tp = sum([metrics[cls]['tp'] for cls in metrics])
        total_fp = sum([metrics[cls]['fp'] for cls in metrics])
        total_fn = sum([metrics[cls]['fn'] for cls in metrics])
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        metrics['overall'] = {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1
        }
        return metrics

    def save_confusion_matrix_csv(self, save_path):
        if self.confusion_matrix is None:
            raise ValueError("Confusion matrix not computed. Run process_files() first.")
        import pandas as pd
        if self.class_names and len(self.class_names) >= len(self.classes):
            labels = [self.class_names[i] for i in sorted(self.classes)] + ['background']
        else:
            labels = [f'Class_{i}' for i in sorted(self.classes)] + ['background']
        df = pd.DataFrame(self.confusion_matrix, index=labels, columns=labels)
        df.to_csv(save_path)
        print(f"Confusion matrix saved to {save_path}")

def main():
    # Ganti ini ke folder yang sudah ada berisi txt YOLO
    pred_folder = "femur_fracture_app/Hasil Deteksi/Detection Mode/New"  # Folder prediksi
    gt_folder = "Pengujian/Hasil Anotasi Dokter/User Training/labels"   # Folder ground truth

    # Ubah sesuai daftar kelas Anda
    class_names = ['Greater Trochanter', 'Intertrochanteric', 'Lesser Trochanter', 'Neck', 'Subtrochanteric']

    cm_generator = YOLOConfusionMatrix(
        pred_folder=pred_folder,
        gt_folder=gt_folder,
        class_names=class_names,
        iou_threshold=0.5
    )
    
    print("Memproses file...")
    cm_generator.process_files()
    print("Selesai memproses!")

    print("Menampilkan confusion matrix...")
    cm_generator.plot_confusion_matrix(normalize=False, save_path='confusion_matrix.png')

    print("Menampilkan confusion matrix (dinormalisasi)...")
    cm_generator.plot_confusion_matrix(normalize=True, save_path='confusion_matrix_normalized.png')

    metrics = cm_generator.calculate_metrics()
    print("\n"+"="*80)
    print("Metrik Evaluasi (per kelas)")
    print("="*80)
    print(f'{"Kelas":<15} {"Precision":<10} {"Recall":<10} {"F1-Score":<10} {"TP":<5} {"FP":<5} {"FN":<5}')
    print("-"*80)
    for class_name, metric in metrics.items():
        if class_name != 'overall':
            print(f"{class_name:<15} {metric['precision']:<10.3f} {metric['recall']:<10.3f} "
                  f"{metric['f1_score']:<10.3f} {metric['tp']:<5} {metric['fp']:<5} {metric['fn']:<5}")
    print("-"*80)
    print(f"{'Overall':<15} {metrics['overall']['precision']:<10.3f} {metrics['overall']['recall']:<10.3f} "
          f"{metrics['overall']['f1_score']:<10.3f}")

    cm_generator.save_confusion_matrix_csv('confusion_matrix.csv')

if __name__ == "__main__":
    main()
