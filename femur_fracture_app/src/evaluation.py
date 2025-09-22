import numpy as np

class BoundingBoxEvaluator:
    @staticmethod
    def calculate_intersection_area(box1, box2):
        """
        Menghitung luas iris (intersection) antara dua bounding boxes
        box format: [x1, y1, x2, y2]
        """
        x_left   = max(box1[0], box2[0])
        y_top    = max(box1[1], box2[1])
        x_right  = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right <= x_left or y_bottom <= y_top:
            return 0

        return (x_right - x_left) * (y_bottom - y_top)

    @staticmethod
    def calculate_union_area(box1, box2, intersection_area):
        """
        Menghitung luas union antara dua bounding boxes
        box format: [x1, y1, x2, y2]
        """
        area1 = max(0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        area2 = max(0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
        return area1 + area2 - intersection_area
    
    @staticmethod
    def evaluate_user_annotation(user_boxes, model_boxes, iou_threshold=0.5):
        """
        Evaluasi one-to-one dengan menghitung persentase IoU:
        IoU = intersection_area / union_area.
        Mengembalikan:
        - total_model: jumlah BB model
        - correct_count: jumlah BB model yang IoU >= threshold
        - percentage_correct: correct_count / total_model
        - comparisons: daftar dict tiap model_box dengan fields:
            model_idx, best_iou, match
        """
        total_model = len(model_boxes)
        comparisons = []
        correct_count = 0

        for mi, mb in enumerate(model_boxes):
            best_iou = 0.0

            for ub in user_boxes:
                inter = BoundingBoxEvaluator.calculate_intersection_area(ub, mb)
                union = BoundingBoxEvaluator.calculate_union_area(ub, mb, inter)
                iou = inter / union if union > 0 else 0.0

                if iou > best_iou:
                    best_iou = iou

            match = best_iou >= iou_threshold
            if match:
                correct_count += 1

            comparisons.append({
                'model_idx': mi,
                'best_iou': best_iou,
                'match': match
            })

        percentage_correct = (correct_count / total_model) if total_model > 0 else 0.0

        return {
            'total_model': total_model,
            'correct_count': correct_count,
            'percentage_correct': percentage_correct,
            'comparisons': comparisons
        }
        
    @staticmethod
    def compute_detection_metrics(user_boxes, model_boxes, iou_threshold=0.5):
        """
        Menghitung TP, FP, FN, Precision, Recall.
        """
        # Hitung IoU matrix
        tp = fp = fn = 0
        matched_model = set()
        for ub in user_boxes:
            best_iou = 0.0
            best_mi = None
            for mi, mb in enumerate(model_boxes):
                inter = BoundingBoxEvaluator.calculate_intersection_area(ub, mb)
                union = BoundingBoxEvaluator.calculate_union_area(ub, mb, inter)
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_mi = iou, mi
            if best_iou >= iou_threshold and best_mi not in matched_model:
                tp += 1
                matched_model.add(best_mi)
            else:
                fp += 1
        # FN = model_boxes yang tidak ter-matched
        fn = len(model_boxes) - len(matched_model)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall
        }

class BoxCountErrorEvaluator:
    """
    Evaluator untuk menghitung persentase kesalahan jumlah bounding box
    antara anotasi user dan prediksi model.
    """
    @staticmethod
    def evaluate(user_boxes, model_boxes):
        """
        Args:
            user_boxes (list): daftar bounding box user
            model_boxes (list): daftar bounding box model

        Returns:
            dict: {
                "total_user": int,
                "total_model": int,
                "difference": int,
                "error_percentage": float
            }
        """
        total_user = len(user_boxes)
        total_model = len(model_boxes)
        difference = total_model - total_user
        error_percentage = abs(difference) / total_model if total_model > 0 else 0.0
        accuracy_percentage = 1.0 - error_percentage

        return {
            "total_user": total_user,
            "total_model": total_model,
            "difference": difference,
            "error_percentage": error_percentage,
            "accuracy_percentage": accuracy_percentage
        }