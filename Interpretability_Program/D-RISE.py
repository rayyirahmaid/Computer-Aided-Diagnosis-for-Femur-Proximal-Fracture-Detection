import os
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
from vision_explanation_methods.explanations.drise import DRISE_saliency
from src.d_rise_yolo import YOLOv8Wrapper
from src.visualization import generate_saliency_maps_yolov8
from src.d_rise_modified import DRISE_saliency_with_debug, set_seed
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import json
import cv2
from scipy import ndimage
from sklearn.metrics import auc
import glob


class QuantitativeEvaluator:
    """Class for evaluating saliency maps with quantitative metrics"""
    
    def __init__(self):
        self.results = {
            'deletion_auc': [],
            'insertion_auc': [],
            'iou_scores': [],
            'pointing_game_scores': []
        }
    
    def load_yolo_annotations(self, annotation_path, img_width, img_height):
        """
        Load YOLO format annotations from txt file
        
        Args:
            annotation_path: Path to YOLO annotation file
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            List of bounding boxes in format [x1, y1, x2, y2]
        """
        bboxes = []
        if not os.path.exists(annotation_path):
            return bboxes
            
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                center_x = float(parts[1]) * img_width
                center_y = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                # Convert center coordinates to corner coordinates
                x1 = center_x - width / 2
                y1 = center_y - height / 2
                x2 = center_x + width / 2
                y2 = center_y + height / 2
                
                bboxes.append([x1, y1, x2, y2, class_id])
                
        return bboxes
    
    def create_ground_truth_mask(self, bboxes, img_width, img_height):
        """
        Create binary mask from bounding boxes
        
        Args:
            bboxes: List of bounding boxes [x1, y1, x2, y2, class_id]
            img_width: Image width
            img_height: Image height
            
        Returns:
            Binary mask with 1s inside bounding boxes
        """
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox[:4])
            x1 = max(0, min(x1, img_width-1))
            y1 = max(0, min(y1, img_height-1))
            x2 = max(0, min(x2, img_width-1))
            y2 = max(0, min(y2, img_height-1))
            
            mask[y1:y2+1, x1:x2+1] = 1
            
        return mask
    
    def normalize_saliency_map(self, saliency_map):
        """Normalize saliency map to [0, 1] range"""
        saliency_map = saliency_map.astype(np.float32)
        min_val = np.min(saliency_map)
        max_val = np.max(saliency_map)
        
        if max_val > min_val:
            return (saliency_map - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(saliency_map)
    
    def deletion_auc(self, model, image, saliency_map, target_class, num_steps=50):
        """
        Calculate Deletion AUC metric[1][4]
        
        Args:
            model: YOLOv8 model wrapper
            image: Original image tensor
            saliency_map: Saliency map array
            target_class: Target class for evaluation
            num_steps: Number of deletion steps
            
        Returns:
            AUC score for deletion curve
        """
        saliency_map = self.normalize_saliency_map(saliency_map)
        
        # Get initial confidence
        initial_output = model.model(image)
        if len(initial_output) == 0:
            return 0.0
            
        initial_conf = self._get_max_confidence(initial_output, target_class)
        
        # Create deletion curve
        confidences = [initial_conf]
        
        # Get pixel indices sorted by saliency (highest first)
        flat_saliency = saliency_map.flatten()
        sorted_indices = np.argsort(flat_saliency)[::-1]
        
        h, w = saliency_map.shape
        masked_image = image.clone()
        
        pixels_to_delete = len(sorted_indices) // num_steps
        
        for step in range(1, num_steps + 1):
            # Delete top saliency pixels
            end_idx = min(step * pixels_to_delete, len(sorted_indices))
            indices_to_delete = sorted_indices[:end_idx]
            
            # Convert flat indices to 2D coordinates
            y_coords = indices_to_delete // w
            x_coords = indices_to_delete % w
            
            # Create mask and apply to image
            temp_image = masked_image.clone()
            temp_image[0, :, y_coords, x_coords] = 0.5  # Gray out deleted pixels
            
            # Get confidence after deletion
            output = model.model(temp_image)
            conf = self._get_max_confidence(output, target_class) if len(output) > 0 else 0.0
            confidences.append(conf)
        
        # Calculate AUC
        x_values = np.linspace(0, 1, len(confidences))
        return auc(x_values, confidences)
    
    def insertion_auc(self, model, image, saliency_map, target_class, num_steps=50):
        """
        Calculate Insertion AUC metric[1][4]
        
        Args:
            model: YOLOv8 model wrapper
            image: Original image tensor
            saliency_map: Saliency map array
            target_class: Target class for evaluation
            num_steps: Number of insertion steps
            
        Returns:
            AUC score for insertion curve
        """
        saliency_map = self.normalize_saliency_map(saliency_map)
        
        # Start with blank (gray) image
        h, w = image.shape[2], image.shape[3]
        blank_image = torch.full_like(image, 0.5)
        
        # Get pixel indices sorted by saliency (highest first)
        flat_saliency = saliency_map.flatten()
        sorted_indices = np.argsort(flat_saliency)[::-1]
        
        confidences = []
        pixels_to_insert = len(sorted_indices) // num_steps
        
        for step in range(num_steps + 1):
            if step == 0:
                current_image = blank_image.clone()
            else:
                # Insert top saliency pixels
                end_idx = min(step * pixels_to_insert, len(sorted_indices))
                indices_to_insert = sorted_indices[:end_idx]
                
                # Convert flat indices to 2D coordinates
                y_coords = indices_to_insert // w
                x_coords = indices_to_insert % w
                
                # Insert original pixels
                current_image = blank_image.clone()
                current_image[0, :, y_coords, x_coords] = image[0, :, y_coords, x_coords]
            
            # Get confidence after insertion
            output = model.model(current_image)
            conf = self._get_max_confidence(output, target_class) if len(output) > 0 else 0.0
            confidences.append(conf)
        
        # Calculate AUC
        x_values = np.linspace(0, 1, len(confidences))
        return auc(x_values, confidences)
    
    def calculate_iou(self, saliency_map, ground_truth_mask, threshold=0.5):
        """
        Calculate IoU between saliency map and ground truth[21][27]
        
        Args:
            saliency_map: Normalized saliency map
            ground_truth_mask: Binary ground truth mask
            threshold: Threshold for binarizing saliency map
            
        Returns:
            IoU score
        """
        # Resize saliency map to match ground truth if needed
        if saliency_map.shape != ground_truth_mask.shape:
            saliency_map = cv2.resize(saliency_map, (ground_truth_mask.shape[1], ground_truth_mask.shape[0]))
        
        # Normalize and threshold saliency map
        saliency_map = self.normalize_saliency_map(saliency_map)
        binary_saliency = (saliency_map > threshold).astype(np.uint8)
        
        # Calculate intersection and union
        intersection = np.logical_and(binary_saliency, ground_truth_mask)
        union = np.logical_or(binary_saliency, ground_truth_mask)
        
        if np.sum(union) == 0:
            return 0.0
        
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score
    
    def pointing_game(self, saliency_map, bboxes):
        """
        Calculate Pointing Game accuracy[3][6][9]
        
        Args:
            saliency_map: Saliency map array
            bboxes: List of bounding boxes [x1, y1, x2, y2, class_id]
            
        Returns:
            Pointing game score (1 if hit, 0 if miss)
        """
        if len(bboxes) == 0:
            return 0.0
        
        # Find maximum saliency location
        max_location = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
        max_y, max_x = max_location
        
        # Check if max location falls within any bounding box
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox[:4])
            if x1 <= max_x <= x2 and y1 <= max_y <= y2:
                return 1.0
        
        return 0.0
    
    def _get_max_confidence(self, detections, target_class=None):
        """Get maximum confidence from model detections"""
        if len(detections) == 0:
            return 0.0
        
        max_conf = 0.0
        for detection in detections:
            if hasattr(detection, 'boxes') and len(detection.boxes) > 0:
                confidences = detection.boxes.conf.cpu().numpy()
                if target_class is not None and hasattr(detection.boxes, 'cls'):
                    classes = detection.boxes.cls.cpu().numpy()
                    target_confidences = confidences[classes == target_class]
                    if len(target_confidences) > 0:
                        max_conf = max(max_conf, np.max(target_confidences))
                else:
                    max_conf = max(max_conf, np.max(confidences))
        
        return max_conf
    
    def evaluate_saliency_map(self, model, image_path, saliency_map, annotation_path, target_class=None):
        """
        Comprehensive evaluation of a single saliency map
        
        Args:
            model: YOLOv8 model wrapper
            image_path: Path to original image
            saliency_map: Saliency map to evaluate
            annotation_path: Path to YOLO annotation file
            target_class: Target class for evaluation
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        img_width, img_height = image.size
        
        # Convert to tensor for model evaluation
        image_tensor = ToTensor()(image).unsqueeze(0)
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        
        # Load ground truth annotations
        bboxes = self.load_yolo_annotations(annotation_path, img_width, img_height)
        ground_truth_mask = self.create_ground_truth_mask(bboxes, img_width, img_height)
        
        results = {}
        
        try:
            # Calculate Deletion AUC
            results['deletion_auc'] = self.deletion_auc(model, image_tensor, saliency_map, target_class)
            
            # Calculate Insertion AUC
            results['insertion_auc'] = self.insertion_auc(model, image_tensor, saliency_map, target_class)
            
            # Calculate IoU with Ground Truth
            results['iou_score'] = self.calculate_iou(saliency_map, ground_truth_mask)
            
            # Calculate Pointing Game
            results['pointing_game'] = self.pointing_game(saliency_map, bboxes)
            
        except Exception as e:
            print(f"Error evaluating saliency map for {image_path}: {str(e)}")
            results = {
                'deletion_auc': 0.0,
                'insertion_auc': 0.0,
                'iou_score': 0.0,
                'pointing_game': 0.0
            }
        
        return results
    
    def save_evaluation_results(self, results, output_path):
        """Save evaluation results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary statistics
        print("\n" + "="*60)
        print("QUANTITATIVE EVALUATION RESULTS")
        print("="*60)
        
        for metric_name, values in results.items():
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{metric_name.upper()}: {mean_val:.4f} ± {std_val:.4f}")
        
        print("="*60)


def generate_saliency_maps_yolov8_with_evaluation(
    images_folder_path,
    model_path,
    output_dir,
    annotations_folder_path,  # New parameter for annotations
    conf_threshold=0.30,
    img_size=800,
    mask_num=600,
    mask_res=32,
    mask_padding=None,
    save_masks=True,
    save_masked_images=True,
    save_predictions=True,
    save_individual_saliency=True,
    debug_sample_count=45,
    resolution_decrease_factor=0.5,
    max_resolution_attempts=3,
    mark_high_intensity=False,
    mark_high_intensity_threshold_mid=0.8,
    mark_high_intensity_threshold_high=0.9
):
    """
    Enhanced function that generates saliency maps and evaluates them quantitatively
    """
    
    # Initialize evaluator
    evaluator = QuantitativeEvaluator()
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    evaluation_dir = os.path.join(output_dir, "quantitative_evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)
    
    # Load YOLO model
    print("Loading YOLOv8 model...")
    model = YOLOv8Wrapper(model_path)
    
    # Get list of images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(images_folder_path, ext.upper())))
    
    print(f"Found {len(image_files)} images to process")
    
    all_results = {
        'deletion_auc': [],
        'insertion_auc': [],
        'iou_scores': [],
        'pointing_game_scores': [],
        'image_results': {}
    }
    
    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            image_name = os.path.basename(image_path)
            base_name = os.path.splitext(image_name)[0]
            
            # Find corresponding annotation file
            annotation_path = os.path.join(annotations_folder_path, f"{base_name}.txt")
            
            if not os.path.exists(annotation_path):
                print(f"Warning: No annotation found for {image_name}, skipping evaluation")
                continue
            
            # Generate saliency map using original function
            print(f"Generating saliency map for {image_name}...")
            
            # Here you would call your original saliency generation function
            # For now, I'll assume it generates and saves saliency maps
            generate_saliency_maps_yolov8(
                images_folder_path=images_folder_path,
                model_path=model_path,
                output_dir=output_dir,
                conf_threshold=conf_threshold,
                img_size=img_size,
                mask_num=mask_num,
                mask_res=mask_res,
                mask_padding=mask_padding,
                save_masks=save_masks,
                save_masked_images=save_masked_images,
                save_predictions=save_predictions,
                save_individual_saliency=save_individual_saliency,
                debug_sample_count=debug_sample_count,
                resolution_decrease_factor=resolution_decrease_factor,
                max_resolution_attempts=max_resolution_attempts,
                mark_high_intensity=mark_high_intensity,
                mark_high_intensity_threshold_mid=mark_high_intensity_threshold_mid,
                mark_high_intensity_threshold_high=mark_high_intensity_threshold_high
            )
            
            # Load generated saliency map
            saliency_map_path = os.path.join(output_dir, "saliency_maps", f"{base_name}_saliency.png")
            
            if os.path.exists(saliency_map_path):
                saliency_map = cv2.imread(saliency_map_path, cv2.IMREAD_GRAYSCALE)
                saliency_map = saliency_map.astype(np.float32) / 255.0
                
                # Evaluate saliency map
                print(f"Evaluating saliency map for {image_name}...")
                results = evaluator.evaluate_saliency_map(
                    model, image_path, saliency_map, annotation_path
                )
                
                # Store results
                all_results['deletion_auc'].append(results['deletion_auc'])
                all_results['insertion_auc'].append(results['insertion_auc'])
                all_results['iou_scores'].append(results['iou_score'])
                all_results['pointing_game_scores'].append(results['pointing_game'])
                all_results['image_results'][image_name] = results
                
            else:
                print(f"Warning: Saliency map not found for {image_name}")
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    # Save evaluation results
    results_path = os.path.join(evaluation_dir, "evaluation_results.json")
    evaluator.save_evaluation_results(all_results, results_path)
    
    # Generate evaluation report
    generate_evaluation_report(all_results, evaluation_dir)
    
    print(f"\nQuantitative evaluation completed!")
    print(f"Results saved to: {evaluation_dir}")


def generate_evaluation_report(results, output_dir):
    """Generate comprehensive evaluation report with visualizations"""
    
    # Create summary statistics
    summary = {}
    for metric_name, values in results.items():
        if isinstance(values, list) and len(values) > 0:
            summary[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
    
    # Save summary
    with open(os.path.join(output_dir, "summary_statistics.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Plot deletion and insertion AUC
    plt.subplot(2, 2, 1)
    if len(results['deletion_auc']) > 0:
        plt.hist(results['deletion_auc'], bins=20, alpha=0.7, label='Deletion AUC')
    if len(results['insertion_auc']) > 0:
        plt.hist(results['insertion_auc'], bins=20, alpha=0.7, label='Insertion AUC')
    plt.xlabel('AUC Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of AUC Scores')
    plt.legend()
    
    # Plot IoU scores
    plt.subplot(2, 2, 2)
    if len(results['iou_scores']) > 0:
        plt.hist(results['iou_scores'], bins=20, alpha=0.7, color='green')
        plt.xlabel('IoU Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of IoU Scores')
    
    # Plot Pointing Game scores
    plt.subplot(2, 2, 3)
    if len(results['pointing_game_scores']) > 0:
        pg_scores = results['pointing_game_scores']
        accuracy = np.mean(pg_scores)
        plt.bar(['Miss', 'Hit'], [1-accuracy, accuracy], color=['red', 'green'])
        plt.ylabel('Proportion')
        plt.title(f'Pointing Game Accuracy: {accuracy:.3f}')
    
    # Correlation plot
    plt.subplot(2, 2, 4)
    if len(results['deletion_auc']) > 0 and len(results['iou_scores']) > 0:
        plt.scatter(results['deletion_auc'], results['iou_scores'], alpha=0.6)
        plt.xlabel('Deletion AUC')
        plt.ylabel('IoU Score')  
        plt.title('Deletion AUC vs IoU Correlation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_plots.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation report generated in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run object detection and D-RISE explanation with quantitative evaluation."
    )

    # Positional arguments - now optional with defaults
    parser.add_argument(
        "images_folder_path",
        nargs="?",
        type=str,
        default="Datasets/Uji/N (95)",
        help="Path to folder containing images"
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        type=str,
        default="Model/Femur/best.pt",
        help="Path to YOLOv8 model file (.pt)"
    )
    parser.add_argument(
        "output_folder_path",
        nargs="?",
        type=str,
        default="Output/Visualize BB/Saliency Maps",
        help="Path to output folder for results"
    )
    parser.add_argument(
        "annotations_folder_path",
        nargs="?",
        type=str,
        default="Pengujian/Hasil Anotasi Dokter/labels/N (95)",
        help="Path to folder containing YOLO format annotation files"
    )
    parser.add_argument(
        "img_size",
        nargs="?",
        type=int,
        default=800,
        help="Size (px) to resize each image"
    )

    # Optional arguments for D-RISE parameters
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.30,
        help="Confidence threshold for object detection"
    )
    parser.add_argument(
        "--masks_num",
        type=int,
        default=2000,
        help="Number of masks to generate for D-RISE"
    )
    parser.add_argument(
        "--mask_res",
        type=int,
        default=16,
        help="Base resolution of masks"
    )
    parser.add_argument(
        "--mask_padding",
        type=int,
        default=None,
        help="Padding for masks (if needed)"
    )
    parser.add_argument(
        "--save_masks",
        action="store_true",
        default=True,
        help="Save generated mask files"
    )
    parser.add_argument(
        "--save_masked_images",
        action="store_true",
        default=True,
        help="Save images with masks applied"
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        default=True,
        help="Save prediction results on masked images"
    )
    parser.add_argument(
        "--save_individual_saliency",
        action="store_true",
        default=True,
        help="Save individual saliency maps"
    )
    parser.add_argument(
        "--debug_sample_count",
        type=int,
        default=45,
        help="Number of debug samples to save"
    )
    parser.add_argument(
        "--deterministic_generation",
        action="store_true",
        help="Use fixed seed for reproducibility"
    )
    parser.add_argument(
        "--resolution_decrease_factor",
        type=float,
        default=0.5,
        help="Resolution decrease factor if saliency map is flat"
    )
    parser.add_argument(
        "--max_resolution_attempts",
        type=int,
        default=3,
        help="Maximum attempts to find suitable mask resolution"
    )
    parser.add_argument(
        "--mark_high_intensity",
        action="store_true",
        help="Mark high intensity areas on saliency map"
    )
    parser.add_argument(
        "--mark_high_intensity_threshold_mid",
        type=float,
        default=0.8,
        help="Mid intensity threshold"
    )
    parser.add_argument(
        "--mark_high_intensity_threshold_high",
        type=float,
        default=0.9,
        help="High intensity threshold"
    )

    args = parser.parse_args()

    # Check device
    if torch.cuda.is_available():
        print("Using GPU for processing.")
    else:
        print("Using CPU for processing.")

    # Set seed for reproducibility
    if args.deterministic_generation:
        set_seed(42)
        print("Using deterministic generation.")

    # Check if annotations folder exists
    if not os.path.exists(args.annotations_folder_path):
        print(f"Error: Annotations folder not found: {args.annotations_folder_path}")
        print("Please provide a valid path to your YOLO annotation files.")
        return

    # Run enhanced function with quantitative evaluation
    generate_saliency_maps_yolov8_with_evaluation(
        images_folder_path=args.images_folder_path,
        model_path=args.model_path,
        output_dir=args.output_folder_path,
        annotations_folder_path=args.annotations_folder_path,  # New parameter
        conf_threshold=args.conf_threshold,
        img_size=args.img_size,
        mask_num=args.masks_num,
        mask_res=args.mask_res,
        mask_padding=args.mask_padding,
        save_masks=args.save_masks,
        save_masked_images=args.save_masked_images,
        save_predictions=args.save_predictions,
        save_individual_saliency=args.save_individual_saliency,
        debug_sample_count=args.debug_sample_count,
        resolution_decrease_factor=args.resolution_decrease_factor,
        max_resolution_attempts=args.max_resolution_attempts,
        mark_high_intensity=args.mark_high_intensity,
        mark_high_intensity_threshold_mid=args.mark_high_intensity_threshold_mid,
        mark_high_intensity_threshold_high=args.mark_high_intensity_threshold_high
    )


if __name__ == "__main__":
    main()
