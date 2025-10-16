import os
import cv2
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Configuration: hardcoded folder paths and parameters
# ─────────────────────────────────────────────────────────────────────────────
IMAGES_FOLDER   = "Datasets/Uji"     # Path to images folder
LABELS1_FOLDER  = "Pengujian/Hasil Anotasi Dokter/labels"    # Path to first labels folder
LABELS2_FOLDER  = "femur_fracture_app/Hasil Deteksi/Detection Mode/labels"    # Path to second labels folder (set "" to disable)
OUTPUT_FOLDER   = "Output/Visualize BB/Dokter_Irisan"    # Path to save visualized images (set "" to display)
IMAGE_EXT       = ".png"       # Image file extension to process
CLASS_NAMES     = [            # Class names ordered by class_id
    "Greater Trochanter", "Intertrochanteric", "Lesser Trochanter", "Neck", "Subtrochanteric"
]
BOX_COLOR1      = (0, 255, 0)  # Green for labels1
BOX_COLOR2      = (0, 0, 255)  # Red   for labels2
THICKNESS       = 2            # Bounding box line thickness
FONT_SCALE      = 0.5          # Font scale for labels
FONT_THICKNESS  = 1            # Font thickness for labels
# ─────────────────────────────────────────────────────────────────────────────

def yolo_to_pixels(box: List[float], w: int, h: int) -> Tuple[int,int,int,int]:
    """
    Convert YOLO (x_center, y_center, width, height) normalized coordinates
    to pixel coordinates (xmin, ymin, xmax, ymax).
    """
    x_ctr, y_ctr, bw, bh = box
    x_ctr, y_ctr = x_ctr * w, y_ctr * h
    bw, bh = bw * w, bh * h
    xmin = int(x_ctr - bw / 2)
    ymin = int(y_ctr - bh / 2)
    xmax = int(x_ctr + bw / 2)
    ymax = int(y_ctr + bh / 2)
    return xmin, ymin, xmax, ymax

def load_labels(path: str) -> dict:
    """
    Load all .txt label files in `path`.
    Returns a dict mapping image basename → list of (class_id, box_coords).
    """
    labels = {}
    if not path or not os.path.isdir(path):
        return labels
    for fn in os.listdir(path):
        if not fn.lower().endswith(".txt"):
            continue
        base = os.path.splitext(fn)[0]
        full = os.path.join(path, fn)
        with open(full, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        boxes = []
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            boxes.append((cls_id, coords))
        labels[base] = boxes
    return labels

def draw_boxes(img, boxes: List[Tuple[int,List[float]]],
               class_names: List[str], color: Tuple[int,int,int]):
    """
    Draw bounding boxes and class labels on the image.
    """
    h, w = img.shape[:2]
    for cls_id, box in boxes:
        xmin, ymin, xmax, ymax = yolo_to_pixels(box, w, h)
        # Draw rectangle
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, THICKNESS)
        # Prepare label background
        label_text = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX,
                                      FONT_SCALE, FONT_THICKNESS)
        cv2.rectangle(img, (xmin, ymin - th - 4), (xmin + tw, ymin), color, -1)
        cv2.putText(img, label_text, (xmin, ymin - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255),
                    FONT_THICKNESS)

def visualize_all():
    """
    Process all images in IMAGES_FOLDER: overlay bounding boxes
    loaded from LABELS1_FOLDER and LABELS2_FOLDER, then display
    or save to OUTPUT_FOLDER.
    """
    # Prepare output folder
    if OUTPUT_FOLDER:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # List all images with the given extension
    images = [f for f in os.listdir(IMAGES_FOLDER)
              if f.lower().endswith(IMAGE_EXT)]

    # Load labels
    labels1 = load_labels(LABELS1_FOLDER)
    labels2 = load_labels(LABELS2_FOLDER)

    for img_file in images:
        base = os.path.splitext(img_file)[0]
        img_path = os.path.join(IMAGES_FOLDER, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Cannot load image {img_path}, skipping.")
            continue

        # Draw from first labels
        if base in labels1:
            draw_boxes(img, labels1[base], CLASS_NAMES, BOX_COLOR1)

        # Draw from second labels
        if LABELS2_FOLDER and base in labels2:
            draw_boxes(img, labels2[base], CLASS_NAMES, BOX_COLOR2)

        # Display or save
        if OUTPUT_FOLDER:
            out_path = os.path.join(OUTPUT_FOLDER, img_file)
            cv2.imwrite(out_path, img)
            print(f"[INFO] Saved visualized image to {out_path}")
        else:
            cv2.imshow("YOLO Visualization", img)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC to quit
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    visualize_all()
