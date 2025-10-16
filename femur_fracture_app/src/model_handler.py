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

    def predict(self, image, confidence_threshold=0.5):
        """
        Melakukan prediksi pada gambar X-ray dan menyederhanakan BB
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

        # Untuk tiap kelas, jika >1 BB: ambil irisan semua BB
        for cls, items in grouped.items():
            if len(items) == 1:
                item = items[0]
                x1, y1, x2, y2 = item['box']
                simplified['boxes'].append([x1, y1, x2, y2])
                simplified['confidences'].append(item['conf'])
                simplified['class_names'].append(item['name'])
            else:
                # Hitung irisan dari semua box
                x1_int = max(item['box'][0] for item in items)
                y1_int = max(item['box'][1] for item in items)
                x2_int = min(item['box'][2] for item in items)
                y2_int = min(item['box'][3] for item in items)
                # Pastikan irisan valid
                if x2_int > x1_int and y2_int > y1_int:
                    simplified['boxes'].append([x1_int, y1_int, x2_int, y2_int])
                    avg_conf = float(np.mean([item['conf'] for item in items]))
                    simplified['confidences'].append(avg_conf)
                    simplified['class_names'].append(items[0]['name'])
                else:
                    # Fallback ke union jika tidak ada overlap
                    ux1 = min(item['box'][0] for item in items)
                    uy1 = min(item['box'][1] for item in items)
                    ux2 = max(item['box'][2] for item in items)
                    uy2 = max(item['box'][3] for item in items)
                    simplified['boxes'].append([ux1, uy1, ux2, uy2])
                    avg_conf = float(np.mean([item['conf'] for item in items]))
                    simplified['confidences'].append(avg_conf)
                    simplified['class_names'].append(items[0]['name'])

        # Gambar semua BB hasil simplifikasi
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