import os
import shutil
import cv2

def apply_clahe(dataset_dir, output_dir, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to all images in dataset_dir/images and save results to output_dir/images.
    Copy labels from dataset_dir/labels to output_dir/labels with "_CL" suffix.
    """
    # Create output directories
    images_in = os.path.join(dataset_dir, 'images')
    labels_in = os.path.join(dataset_dir, 'labels')
    images_out = os.path.join(output_dir, 'images')
    labels_out = os.path.join(output_dir, 'labels')
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    # Initialize CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Process each image
    for fname in os.listdir(images_in):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(images_in, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read image {fname}")
            continue

        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L-channel
        cl = clahe.apply(l)

        # Merge channels and convert back to BGR
        merged = cv2.merge((cl, a, b))
        img_clahe = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # Generate new filename with "_CL" suffix
        base, ext = os.path.splitext(fname)
        new_image_fname = f"{base}_CL{ext}"
        out_path = os.path.join(images_out, new_image_fname)
        cv2.imwrite(out_path, img_clahe)

        # Generate corresponding label filename with "_CL" suffix
        original_label_fname = f"{base}.txt"
        new_label_fname = f"{base}_CL.txt"
        src_label = os.path.join(labels_in, original_label_fname)
        dst_label = os.path.join(labels_out, new_label_fname)
        
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            print(f"Warning: Label file not found for {fname}.")

    print(f"Applied CLAHE to images: saved to {output_dir}")

def main():
    dataset_dir = "C:/RayFile/KuliahBro/Semester8/TugasAkhir/Datasets + YOLO code/YOLO_V2DATASET/Proximal-Femur-Fracture-2/train"
    processed_dir = 'C:/RayFile/KuliahBro/Semester8/TugasAkhir/Datasets + YOLO code/CLAHE_V2DATASET/train'
    
    # Apply CLAHE and save to processed_dir
    apply_clahe(dataset_dir, processed_dir)

if __name__ == "__main__":
    main()
