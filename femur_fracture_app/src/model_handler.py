import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


class YOLOModelHandler:
    def __init__(self, model_path):
        """
        Inisialisasi handler untuk model YOLO
        """
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """
        Load model YOLO yang telah dilatih
        """
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
            print(f"Model berhasil dimuat dari {self.model_path}")
        else:
            raise FileNotFoundError(f"Model tidak ditemukan di {self.model_path}")

    def nms_filter(self, boxes, confidences, iou_threshold=0.15):
        """
        Menerapkan Non-Maximum Suppression untuk menghilangkan bounding box yang tumpang tindih
        """
        if len(boxes) == 0:
            return []
        
        # Konversi ke format yang diperlukan oleh cv2.dnn.NMSBoxes
        boxes_array = np.array(boxes, dtype=np.float32)
        confidences_array = np.array(confidences, dtype=np.float32)
        
        # Konversi dari format [x1, y1, x2, y2] ke [x, y, width, height]
        nms_boxes = []
        for box in boxes_array:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            nms_boxes.append([x1, y1, width, height])
        
        nms_boxes = np.array(nms_boxes, dtype=np.float32)
        
        # Terapkan NMS
        indices = cv2.dnn.NMSBoxes(
            nms_boxes.tolist(),
            confidences_array.tolist(),
            score_threshold=0.0,  # Sudah difilter oleh confidence_threshold
            nms_threshold=iou_threshold
        )
        
        # Kembalikan indeks yang terpilih
        if len(indices) > 0:
            return indices.flatten()
        else:
            return []

    def predict(self, image, confidence_threshold=0.5, nms_threshold=0.15):
        """
        Melakukan prediksi pada gambar X-ray dengan NMS untuk menghilangkan duplikasi BB
        """
        if self.model is None:
            raise ValueError("Model belum dimuat dengan benar")

        # Konversi PIL Image ke array
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image.copy()
            
        # Tentukan format asli: grayscale (2D) atau RGB (3D)
        original_format = image_array.ndim, image_array.shape[-1] if image_array.ndim == 3 else 1
        
        # Jika array 2D (grayscale), ubah ke BGR
        if image_array.ndim == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        # Jika array 3D tapi hanya 1 channel, juga ubah
        elif image_array.ndim == 3 and image_array.shape[2] == 1:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        
        # Jalankan prediksi
        results = self.model(image_array, conf=confidence_threshold)[0]

        # Ekstrak boxes, confidences, kelas, dan nama
        boxes = results.boxes.xyxy.cpu().numpy()        # shape (N,4)
        confidences = results.boxes.conf.cpu().numpy()  # shape (N,)
        classes = results.boxes.cls.cpu().numpy().astype(int)  # shape (N,)
        names = [results.names[c] for c in classes]

        # Kelompokkan BB berdasarkan kelas
        grouped = {}
        for box, conf, cls, name in zip(boxes, confidences, classes, names):
            grouped.setdefault(cls, []).append({
                'box': box.astype(int),
                'conf': float(conf),
                'name': name
            })

        simplified = {
            'boxes': [],
            'confidences': [],
            'class_names': [],
            'annotated_image': image_array.copy()
        }

        # Untuk tiap kelas, terapkan NMS jika >1 BB
        for cls, items in grouped.items():
            if len(items) == 1:
                # Hanya satu bounding box, langsung tambahkan
                item = items[0]
                x1, y1, x2, y2 = item['box']
                simplified['boxes'].append([x1, y1, x2, y2])
                simplified['confidences'].append(item['conf'])
                simplified['class_names'].append(item['name'])
            else:
                # Lebih dari satu bounding box, terapkan NMS
                class_boxes = [item['box'] for item in items]
                class_confs = [item['conf'] for item in items]
                
                # Terapkan NMS
                selected_indices = self.nms_filter(
                    class_boxes, 
                    class_confs, 
                    iou_threshold=nms_threshold
                )
                
                # Tambahkan bounding box yang terpilih setelah NMS
                for idx in selected_indices:
                    item = items[idx]
                    x1, y1, x2, y2 = item['box']
                    simplified['boxes'].append([x1, y1, x2, y2])
                    simplified['confidences'].append(item['conf'])
                    simplified['class_names'].append(item['name'])

        # Gambar semua BB hasil NMS
        for (x1, y1, x2, y2), conf, name in zip(
            simplified['boxes'],
            simplified['confidences'],
            simplified['class_names']
        ):
            cv2.rectangle(
                simplified['annotated_image'],
                (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )
            label = f"{name}: {conf:.2f}"
            cv2.putText(
                simplified['annotated_image'],
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        return simplified