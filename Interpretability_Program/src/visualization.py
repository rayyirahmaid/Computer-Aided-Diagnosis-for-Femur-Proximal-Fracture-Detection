import os
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
from vision_explanation_methods.explanations.drise import DRISE_saliency
from src.d_rise_yolo import YOLOv8Wrapper
from src.d_rise_modified import DRISE_saliency_with_debug, set_seed
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Dict, Any, Tuple, Optional


def save_saliency_images(
        image: Image.Image,
        saliency_map: np.ndarray,
        bbox: np.ndarray,
        score: float,
        label: str,
        output_dir: str,
        image_name: str,
        index: int,
        original_size: tuple = None,
        mark_high_intensity: bool = True,  # Whether to mark high intensity areas
        mark_high_intensity_threshold_mid: float = 0.8,  # Threshold for mid intensity
        mark_high_intensity_threshold_high: float = 0.9,  # Threshold for high intensity
        ):
    """
    Save the original image, saliency mask, and merged saliency mask.
    
    Args:
        image (Image.Image): The original image.
        saliency_map (np.ndarray): The saliency map.
        bbox (np.ndarray): The bounding box coordinates.
        score (float): The confidence score.
        label (str): The class label.
        output_dir (str): The directory to save the images.
        image_name (str): The name of the image file.
        index (int): The index of the detection.
        original_size (tuple): The original size of the image. 
        mark_high_intensity (bool): Whether to mark high intensity areas.
        mark_high_intensity_threshold_mid (float): Threshold for mid intensity.
        mark_high_intensity_threshold_high (float): Threshold for high intensity.   
    """
    os.makedirs(output_dir, exist_ok=True)

    output_dir_overlays = os.path.join(output_dir, "overlays")
    os.makedirs(output_dir_overlays, exist_ok=True)
    
    # Extract saliency mask
    saliency_mask = saliency_map['detection'].cpu().numpy().transpose(1, 2, 0)
    saliency_mask = np.squeeze(saliency_mask)  # Ensure it's 2D for colormap
    if saliency_mask.ndim > 2:
        saliency_mask = saliency_mask[:, :, 0]  # Use only the first channel
    
    # Find the location of the maximum saliency value
    max_y, max_x = np.unravel_index(np.argmax(saliency_mask), saliency_mask.shape)
    max_val = np.max(saliency_mask)
    
    # Create a visualization with matplotlib
    plt.figure(figsize=(15, 5))
    
    # Original image with bounding boxes
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title(f"Original Image - {label}")
    plt.axis('off')
    
    # Draw bounding box for this detection
    x, y, x2, y2 = bbox
    width, height = x2 - x, y2 - y
    plt.gca().add_patch(Rectangle((x, y), width, height, fill=False, edgecolor='red', linewidth=2))
    plt.gca().text(x, y - 10, f"{label} - {score:.2f}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    # Overlay saliency map on original image with highlighted max area
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    
    saliency_overlay = plt.imshow(saliency_mask, alpha=0.3, cmap='jet')
    
    if mark_high_intensity:
        # Mark the point of maximum saliency with a white circle
        #plt.plot(max_x, max_y, 'o', markersize=10, markerfacecolor='none', markeredgecolor='red', markeredgewidth=2)
        
        # Add contours to highlight high-saliency regions
        mid_threshold = mark_high_intensity_threshold_mid * max_val
        high_threshold = mark_high_intensity_threshold_high * max_val
        
        plt.contour(saliency_mask, levels=[mid_threshold], colors=['yellow'], linewidths=1)
        plt.contour(saliency_mask, levels=[high_threshold], colors=['red'], linewidths=1)
    
    plt.colorbar(saliency_overlay, label='Saliency Score', orientation='horizontal', pad=0.1)
    plt.title(f"Overlay - {label}\nMax saliency: {max_val:.3f}")
    plt.axis('off')

    # New figure for saliency overlay mask only
    plt.figure(frameon=False, figsize=(original_size[0] / 100, original_size[1] / 100))
    plt.imshow(image.resize(original_size))  
    saliency_mask_resized = Image.fromarray((saliency_mask * 255).astype(np.uint8)).resize(original_size)  
    plt.imshow(saliency_mask_resized, alpha=0.3, cmap='jet')
    
    if mark_high_intensity:    
        # Add contours to the resized overlay as well
        saliency_mask_np = np.array(saliency_mask_resized)
        max_val_resized = np.max(saliency_mask_np)
        if max_val_resized > 0:  # Prevent division by zero
            plt.contour(saliency_mask_np, levels=[mid_threshold * max_val_resized, high_threshold * max_val_resized], 
                    colors=['yellow', 'red'], linewidths=[1, 2])
    
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.axis('off')
    
    output_file_name_saliency = os.path.basename(image_name)
    output_file_name_saliency = os.path.splitext(output_file_name_saliency)[0]
    output_file_name_saliency = f"{output_file_name_saliency}_{label}_{index}_saliency.png"
    output_file_path_saliency = os.path.join(output_dir_overlays, output_file_name_saliency)
    plt.savefig(output_file_path_saliency, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Cropped image with saliency overlay and highlighted max area
    plt.subplot(1, 3, 3)
    cropped_saliency = saliency_mask[int(y):int(y2), int(x):int(x2)]  # Crop the saliency map to the bounding box
    image_array = np.array(image)  # Convert PIL Image to NumPy array
    cropped_image = image_array[int(y):int(y2), int(x):int(x2)]  # Crop the original image to the bounding box
    
    plt.imshow(cropped_image)
    cropped_overlay = plt.imshow(cropped_saliency, alpha=0.3, cmap='jet')
    
    if mark_high_intensity:
        # Add contours to highlight high-saliency regions in the cropped view
        if np.max(cropped_saliency) > 0:  # Prevent division by zero
            crop_max = np.max(cropped_saliency)
            plt.contour(cropped_saliency, levels=[mid_threshold * crop_max, high_threshold * crop_max], 
                    colors=['yellow', 'red'], linewidths=[1, 2])
        
            # Mark the point of maximum saliency in the cropped view
            crop_max_y, crop_max_x = np.unravel_index(np.argmax(cropped_saliency), cropped_saliency.shape)
            #plt.plot(crop_max_x, crop_max_y, 'o', markersize=10, markerfacecolor='none', 
            #        markeredgecolor='red', markeredgewidth=2)
    
    plt.colorbar(cropped_overlay, label='Saliency Score', orientation='horizontal', pad=0.1)
    plt.title(f"Cropped Overlay - {label}")
    plt.axis('off')

    # Save the visualization in the output folder
    plt.tight_layout()
    output_file_name = os.path.basename(image_name)
    output_file_name = os.path.splitext(output_file_name)[0]  # Remove file extension
    output_file_name = f"{output_file_name}_{label}_{index}.png"  # Append index to filename
    output_file_path = os.path.join(output_dir, output_file_name)
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close()


def generate_saliency_maps_yolov8(
        images_folder_path: str, 
        model_path: str, 
        output_dir: str, 
        conf_threshold: float = 0.5,
        img_size: int = 640,
        mask_num: int = 500,
        mask_res: int = 8,
        mask_padding: int = None,
        save_masks: bool = False,
        save_masked_images: bool = False,
        save_predictions: bool = True,
        save_individual_saliency: bool = True,
        debug_sample_count: int = 10,  # Number of debug samples to save
        max_resolution_attempts: int = 3,  # Maximum number of resolution attempts
        resolution_decrease_factor: float = 0.5,  # Factor to decrease resolution by
        mark_high_intensity: bool = True,  # Whether to mark high intensity areas
        mark_high_intensity_threshold_mid: float = 0.8,  # Threshold for mid intensity
        mark_high_intensity_threshold_high: float = 0.9,  # Threshold for high intensity
        ):
    """
    Generate saliency maps for a YOLOv8 model with extended debug options.

    Args:
        images_folder_path (str): Path to the folder containing images.
        model_path (str): Path to the YOLOv8 model file.
        output_dir (str): Path to the folder where results will be saved.
        conf_threshold (float): Confidence threshold for displaying detections.
        img_size (int): Size to resize images for the model.
        mask_num (int): Number of masks to generate for D-RISE. More is slower but gives higher quality mask.
        mask_res (int): Resolution of the base mask. High resolutions will give finer masks, but more need to be run.
        mask_padding (int): Padding for the mask.
        save_masks (bool): Whether to save generated masks.
        save_masked_images (bool): Whether to save images with masks applied.
        save_predictions (bool): Whether to save prediction results on masked images.
        save_individual_saliency (bool): Whether to save individual saliency maps.
        debug_sample_count (int): Number of debug samples to save per image (to avoid saving all masks).
        max_resolution_attempts (int): Maximum number of attempts to find a suitable mask resolution.
        resolution_decrease_factor (float): Factor to decrease resolution by if flat saliency maps are detected.
        mark_high_intensity (bool): Whether to mark high intensity areas in the saliency maps.
        mark_high_intensity_threshold_mid (float): Threshold for mid intensity.
        mark_high_intensity_threshold_high (float): Threshold for high intensity.

    """
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # if save_masks:
    #     masks_dir = os.path.join(output_dir, "masks")
    #     os.makedirs(masks_dir, exist_ok=True)
    
    # if save_masked_images:
    #     masked_images_dir = os.path.join(output_dir, "masked_images")
    #     os.makedirs(masked_images_dir, exist_ok=True)
    
    # if save_predictions:
    #     predictions_dir = os.path.join(output_dir, "predictions")
    #     os.makedirs(predictions_dir, exist_ok=True)
    
    if save_individual_saliency:
        individual_saliency_dir = os.path.join(output_dir, "individual_saliency_matrix")
        os.makedirs(individual_saliency_dir, exist_ok=True)
    
    # Load the YOLOv8 model
    yolo_wrapper = YOLOv8Wrapper(model_path=model_path)

    # Get the list of image files in the folder and its subfolders
    image_files = []
    for root, _, files in os.walk(images_folder_path):
        image_files.extend([os.path.join(root, f) for f in files if f.endswith(('.jpg', '.png', '.jpeg'))])

    print(f"Found {len(image_files)} images in {images_folder_path}.")

    # Process each image file
    for image_file in tqdm(image_files, desc="Processing images"):
        base_filename = os.path.splitext(os.path.basename(image_file))[0]
        # Explicitly free memory after each image
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        image = Image.open(image_file).convert("RGB")
        if image is None:
            print(f"Failed to load image {image_file}")
            continue
        
        original_size = image.size

        image = image.resize((img_size, img_size))  # Resize to model input size
        image_tensor = ToTensor()(image).unsqueeze(0).to(device="cuda" if torch.cuda.is_available() else "cpu")  # Add batch dimension and move to GPU

        detections = yolo_wrapper.predict(image_tensor, conf_threshold=conf_threshold)
        if not detections or len(detections[0].bounding_boxes) == 0:
            print(f"No detections found for image: {image_file}")
            del image_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            continue

        # Try different resolutions if needed
        current_mask_res = mask_res
        attempt = 0
        successful = False
        
        while attempt < max_resolution_attempts and not successful:
            print(f"Attempt {attempt+1}: Using mask resolution {current_mask_res}")
            
            # Generate saliency maps using D-RISE with debug info
            debug_results = DRISE_saliency_with_debug(
                model=yolo_wrapper,
                image_tensor=image_tensor,
                target_detections=detections,
                number_of_masks=mask_num,
                mask_res=(current_mask_res, current_mask_res),
                mask_padding=mask_padding,
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=True,
                keep_all_debug_data=True,
                max_debug_masks=debug_sample_count,
            )
            
            # Extract saliency maps from debug results
            saliency_maps = debug_results['saliency_maps']
            
            # Check if saliency maps are flat
            flat_saliency_detected = False
            
            # Check the structure of saliency_maps to ensure proper access
            if isinstance(saliency_maps, list) and len(saliency_maps) > 0:
                for i, saliency_map in enumerate(saliency_maps[0]):
                    # Make sure saliency_map is a numpy array before processing
                    if isinstance(saliency_map, dict):
                        # If it's a dictionary, we need to extract the actual map data
                        # Determine the correct key to use (depends on your implementation)
                        if 'map' in saliency_map:
                            saliency_data = saliency_map['map']
                        elif 'saliency' in saliency_map:
                            saliency_data = saliency_map['saliency']
                        elif 'detection' in saliency_map:
                            saliency_data = saliency_map['detection']
                        else:
                            print(f"Warning: Cannot extract saliency data from dictionary for map {i}")
                            print(f"Dictionary keys: {saliency_map.keys()}")
                            # Try to use the first value in the dictionary as a fallback
                            try:
                                saliency_data = next(iter(saliency_map.values()))
                                print(f"Using first available value with type: {type(saliency_data)}")
                            except:
                                continue  # Skip this saliency map if we can't extract data
                    elif hasattr(saliency_map, 'numpy') and callable(getattr(saliency_map, 'numpy')):
                        # If it's a tensor, convert to numpy
                        saliency_data = saliency_map.numpy()
                    else:
                        # Assume it's already a numpy array
                        saliency_data = saliency_map
                    
                    # Make sure saliency_data is a numpy array before computing statistics
                    if hasattr(saliency_data, 'numpy') and callable(getattr(saliency_data, 'numpy')):
                        saliency_data = saliency_data.numpy()
                    
                    if not isinstance(saliency_data, np.ndarray):
                        print(f"Warning: Skipping non-array saliency data of type {type(saliency_data)}")
                        continue
                        
                    # Calculate statistics on the numpy array
                    std_dev = np.std(saliency_data)
                    min_val = np.min(saliency_data)
                    max_val = np.max(saliency_data)
                    range_val = max_val - min_val
                    
                    # Print statistics for debugging
                    print(f"Saliency map {i} statistics: std={std_dev:.6f}, min={min_val:.6f}, max={max_val:.6f}, range={range_val:.6f}")
                    
                    # Define a threshold for "flatness"
                    if std_dev < 0.01 or range_val < 0.25 or np.isnan(std_dev) or np.isnan(std_dev):
                        flat_saliency_detected = True
                        print(f"Flat saliency map detected for detection {i} in {image_file}")
        

            # If flat saliency maps are detected, try a lower resolution
            if flat_saliency_detected and attempt < max_resolution_attempts - 1:
                # Decrease resolution for next attempt
                current_mask_res = max(2, int(current_mask_res * resolution_decrease_factor))
                print(f"Detected flat saliency maps. Decreasing mask resolution to {current_mask_res} for next attempt")
                # Free memory before next attempt
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                attempt += 1
            else:
                successful = True  # Either we found good saliency maps or we've reached max attempts
        
        
        # Save debug outputs
        if any([save_masks, save_masked_images, save_predictions, save_individual_saliency]):
            # Get the actual number of available debug data for each type
            num_available_predictions = len(debug_results['masked_predictions'])
            num_available_masks = len(debug_results.get('masks', []))
            num_available_masked_images = len(debug_results.get('masked_images', []))
            num_available_mask_records = len(debug_results.get('mask_records', []))
            
            print(f"Number of available predictions: {num_available_predictions}")
            print(f"Number of available masks: {num_available_masks}")
            print(f"Number of available masked images: {num_available_masked_images}")
            print(f"Number of available mask records: {num_available_mask_records}")
            
            # Use the minimum available data to ensure we don't go out of bounds
            max_available_data = min(
                num_available_predictions,
                num_available_masks if save_masks else float('inf'),
                num_available_masked_images if save_masked_images else float('inf'),
                num_available_mask_records if save_individual_saliency else float('inf')
            )
            
            if max_available_data == float('inf'):
                max_available_data = num_available_predictions
            
            # Generate sample indices based on the actual available data
            sample_indices = np.linspace(
                0, 
                max_available_data-1, 
                min(debug_sample_count, max_available_data), 
                dtype=int
            )
            
            # for idx in tqdm(sample_indices, desc="Saving debug outputs", total=len(sample_indices)):
            #     # Save masks
            #     if save_masks and idx < num_available_masks:
            #         # Handle tuple format (index, mask)
            #         mask_data = debug_results['masks'][idx]
            #         if isinstance(mask_data, tuple):
            #             mask_idx, mask = mask_data  # Unpack the tuple
            #         else:
            #             mask = mask_data  # If it's not a tuple, use as is
                        
            #         # Handle channel dimension properly - masks should be 1-channel (grayscale)
            #         mask_np = mask.squeeze().cpu().numpy()
            #         # If mask has 3 dimensions, take mean across channels to get 2D
            #         if len(mask_np.shape) == 3 and mask_np.shape[0] == 3:  # (3, H, W) format
            #             mask_np = np.mean(mask_np, axis=0)  # Average across channels
                    
            #         plt.figure(figsize=(8, 8))
            #         plt.imshow(mask_np, cmap='gray')
            #         plt.title(f"Mask {idx}")
            #         plt.colorbar()
            #         plt.savefig(os.path.join(masks_dir, f"{base_filename}_mask_{idx}.png"))
            #         plt.close()
                
            #     # Save masked images
            #     if save_masked_images and idx < num_available_masked_images:
            #         # Handle tuple format (index, masked_image)
            #         masked_img_data = debug_results['masked_images'][idx]
            #         if isinstance(masked_img_data, tuple):
            #             masked_img_idx, masked_img = masked_img_data  # Unpack the tuple
            #         else:
            #             masked_img = masked_img_data  # If it's not a tuple, use as is
                        
            #         # Convert tensor to PIL image format (C,H,W) -> (H,W,C)
            #         masked_img_np = masked_img.squeeze().permute(1, 2, 0).cpu().numpy()
            #         plt.figure(figsize=(8, 8))
            #         plt.imshow(masked_img_np)
            #         plt.title(f"Masked Image {idx}")
            #         plt.savefig(os.path.join(masked_images_dir, f"{base_filename}_masked_{idx}.png"))
            #         plt.close()
                
            #     # Save predictions on masked images
            #     if save_predictions and idx < num_available_predictions and idx < num_available_masked_images:
            #         # Get the masked image and its predictions
            #         masked_img_data = debug_results['masked_images'][idx]
            #         if isinstance(masked_img_data, tuple):
            #             masked_img_idx, masked_img = masked_img_data
            #         else:
            #             masked_img = masked_img_data
                    
            #         pred_data = debug_results['masked_predictions'][idx]
            #         if isinstance(pred_data, tuple):
            #             pred_idx, predictions = pred_data
            #         else:
            #             predictions = pred_data
                    
            #         # Convert tensor to numpy
            #         masked_img_np = masked_img.squeeze().permute(1, 2, 0).cpu().numpy()
                    
            #         # Create visualization
            #         plt.figure(figsize=(10, 10))
            #         plt.imshow(masked_img_np)
                    
            #         # Draw predictions if available
            #         if predictions and len(predictions) > 0:
            #             for pred_idx, prediction in enumerate(predictions):
            #                 if hasattr(prediction, 'bounding_boxes') and prediction.bounding_boxes.shape[0] > 0:
            #                     for box_idx, bbox in enumerate(prediction.bounding_boxes):
            #                         x1, y1, x2, y2 = bbox.cpu().numpy()
            #                         width, height = x2 - x1, y2 - y1
            #                         score = prediction.objectness_scores[box_idx].item()
            #                         label_idx = torch.argmax(prediction.class_scores[box_idx]).item()
            #                         label = yolo_wrapper.model.model.names[label_idx]
                                    
            #                         # Draw box
            #                         rect = plt.Rectangle((x1, y1), width, height, 
            #                                             fill=False, edgecolor='red', linewidth=2)
            #                         plt.gca().add_patch(rect)
            #                         plt.text(x1, y1-10, f"{label}: {score:.2f}", 
            #                                  color='red', fontsize=12, 
            #                                  bbox=dict(facecolor='white', alpha=0.7))
                    
            #         plt.title(f"Predictions on Masked Image {idx}")
            #         plt.savefig(os.path.join(predictions_dir, f"{base_filename}_predictions_{idx}.png"))
            #         plt.close()
            
            # Save individual saliency maps (before fusion)
            if save_individual_saliency and num_available_mask_records > 0:
                print(f"Saving individual saliency maps for {base_filename}")
                for detection_idx in range(len(detections[0].bounding_boxes)):
                    # Get the bounding box info
                    bbox = detections[0].bounding_boxes[detection_idx].cpu().numpy()
                    score = detections[0].objectness_scores[detection_idx].item()
                    label_idx = torch.argmax(detections[0].class_scores[detection_idx]).item()
                    label = yolo_wrapper.model.model.names[label_idx]
                    
                    # Create a figure showing individual saliency contribution
                    plt.figure(figsize=(20, 20))
                    
                    # Use indices that are valid for mask_records
                    valid_indices = [idx for idx in sample_indices if idx < num_available_mask_records]
                    
                    for i, idx in enumerate(valid_indices):
                        # Dynamically determine grid size based on number of valid_indices
                        n = len(valid_indices)
                        if n == 0:
                            continue
                        cols = int(np.ceil(np.sqrt(n)))
                        rows = int(np.ceil(n / cols))

                        if i >= n:
                            break

                        plt.subplot(rows, cols, i + 1)
                        
                        # Get mask and score for this detection
                        mask_record = debug_results['mask_records'][idx]
                        if isinstance(mask_record, tuple):
                            mask_idx, mask_record = mask_record
                            
                        mask = mask_record.mask
                        affinity_scores = mask_record.affinity_scores
                        
                        # Ensure we have valid scores before trying to access them
                        if (len(affinity_scores) > 0 and 
                            detection_idx < len(affinity_scores[0])):
                            
                            # Get the score for this detection
                            detection_score = affinity_scores[0][detection_idx]
                            
                            # Check if score is a tensor or list
                            if hasattr(detection_score, 'dim') and detection_score.dim() == 0:
                                score_value = detection_score.item()
                            elif isinstance(detection_score, (list, tuple)) and len(detection_score) > 0:
                                score_value = detection_score[0]
                            else:
                                score_value = 0
                            
                            # Display the mask with the contribution level
                            mask_np = mask.squeeze().cpu().numpy()
                            if len(mask_np.shape) == 3 and mask_np.shape[0] == 3:
                                mask_np = np.mean(mask_np, axis=0)  # Average across channels if needed
                            
                            plt.imshow(mask_np, cmap='jet')
                            plt.title(f"Score: {score_value:.3f}")
                            plt.axis('off')
                        else:
                            plt.text(0.5, 0.5, "No data", 
                                    horizontalalignment='center',
                                    verticalalignment='center')
                            plt.axis('off')
                    
                    plt.suptitle(f"Individual Saliency Contributions for {label}", fontsize=16)
                    plt.tight_layout()
                    plt.savefig(os.path.join(individual_saliency_dir, 
                                            f"{base_filename}_{label}_{detection_idx}_saliency_components.png"))
                    plt.close()

            
            # Create predictions matrix plot
            if save_predictions and num_available_predictions > 0:
                print(f"Saving predictions matrix for {base_filename}")
                predictions_matrix_dir = os.path.join(output_dir, "predictions_matrix")
                os.makedirs(predictions_matrix_dir, exist_ok=True)
                
                plt.figure(figsize=(20, 20))
                
                # Use sample indices that are valid for predictions
                valid_prediction_indices = [idx for idx in sample_indices if idx < num_available_predictions and idx < num_available_masked_images]
                
                n = len(valid_prediction_indices)
                if n > 0:
                    cols = int(np.ceil(np.sqrt(n)))
                    rows = int(np.ceil(n / cols))
                    
                    for i, idx in enumerate(valid_prediction_indices):
                        if i >= n:
                            break
                            
                        plt.subplot(rows, cols, i + 1)
                        
                        # Get the masked image and its predictions
                        masked_img_data = debug_results['masked_images'][idx]
                        if isinstance(masked_img_data, tuple):
                            masked_img_idx, masked_img = masked_img_data
                        else:
                            masked_img = masked_img_data
                        
                        pred_data = debug_results['masked_predictions'][idx]
                        if isinstance(pred_data, tuple):
                            pred_idx, predictions = pred_data
                        else:
                            predictions = pred_data
                        
                        # Convert tensor to numpy
                        masked_img_np = masked_img.squeeze().permute(1, 2, 0).cpu().numpy()
                        
                        # Display the image
                        plt.imshow(masked_img_np)
                        
                        # Draw predictions if available
                        detection_count = 0
                        if predictions and len(predictions) > 0:
                            for pred_idx, prediction in enumerate(predictions):
                                if hasattr(prediction, 'bounding_boxes') and prediction.bounding_boxes.shape[0] > 0:
                                    for box_idx, bbox in enumerate(prediction.bounding_boxes):
                                        x1, y1, x2, y2 = bbox.cpu().numpy()
                                        width, height = x2 - x1, y2 - y1
                                        score = prediction.objectness_scores[box_idx].item()
                                        label_idx = torch.argmax(prediction.class_scores[box_idx]).item()
                                        label = yolo_wrapper.model.model.names[label_idx]
                                        
                                        # Draw box
                                        rect = plt.Rectangle((x1, y1), width, height, 
                                                            fill=False, edgecolor='red', linewidth=1)
                                        plt.gca().add_patch(rect)

                                        # Add label and confidence score text
                                        plt.text(x1, y1-2, f"{label}: {score:.2f}", 
                                                color='red', fontsize=6, weight='bold',
                                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', linewidth=0.5))

                                        detection_count += 1
                        
                        plt.title(f"Mask {idx}: {detection_count} detections", fontsize=8)
                        plt.axis('off')
                    
                    plt.suptitle("Predictions on All Masked Images", fontsize=16)
                    plt.tight_layout()
                    plt.savefig(os.path.join(predictions_matrix_dir, f"{base_filename}_all_predictions_matrix.png"), dpi=150)
                    plt.close()

            # Create masks matrix plot
            if save_masks and num_available_masks > 0:
                print(f"Saving masks matrix for {base_filename}")
                masks_matrix_dir = os.path.join(output_dir, "masks_matrix")
                os.makedirs(masks_matrix_dir, exist_ok=True)
                
                plt.figure(figsize=(20, 20))
                
                # Use sample indices that are valid for masks
                valid_mask_indices = [idx for idx in sample_indices if idx < num_available_masks]
                
                n = len(valid_mask_indices)
                if n > 0:
                    cols = int(np.ceil(np.sqrt(n)))
                    rows = int(np.ceil(n / cols))
                    
                    for i, idx in enumerate(valid_mask_indices):
                        if i >= n:
                            break
                            
                        plt.subplot(rows, cols, i + 1)
                        
                        # Handle tuple format (index, mask)
                        mask_data = debug_results['masks'][idx]
                        if isinstance(mask_data, tuple):
                            mask_idx, mask = mask_data
                        else:
                            mask = mask_data
                            
                        # Handle channel dimension properly
                        mask_np = mask.squeeze().cpu().numpy()
                        if len(mask_np.shape) == 3 and mask_np.shape[0] == 3:
                            mask_np = np.mean(mask_np, axis=0)
                        
                        plt.imshow(mask_np, cmap='gray')
                        plt.title(f"Mask {idx}", fontsize=8)
                        plt.axis('off')
                    
                    plt.suptitle("All Generated Masks", fontsize=16)
                    plt.tight_layout()
                    plt.savefig(os.path.join(masks_matrix_dir, f"{base_filename}_all_masks_matrix.png"), dpi=150)
                    plt.close()

            # Create masked images matrix plot
            if save_masked_images and num_available_masked_images > 0:
                print(f"Saving masked images matrix for {base_filename}")
                masked_images_matrix_dir = os.path.join(output_dir, "masked_images_matrix")
                os.makedirs(masked_images_matrix_dir, exist_ok=True)
                
                plt.figure(figsize=(20, 20))
                
                # Use sample indices that are valid for masked images
                valid_masked_img_indices = [idx for idx in sample_indices if idx < num_available_masked_images]
                
                n = len(valid_masked_img_indices)
                if n > 0:
                    cols = int(np.ceil(np.sqrt(n)))
                    rows = int(np.ceil(n / cols))
                    
                    for i, idx in enumerate(valid_masked_img_indices):
                        if i >= n:
                            break
                            
                        plt.subplot(rows, cols, i + 1)
                        
                        # Handle tuple format (index, masked_image)
                        masked_img_data = debug_results['masked_images'][idx]
                        if isinstance(masked_img_data, tuple):
                            masked_img_idx, masked_img = masked_img_data
                        else:
                            masked_img = masked_img_data
                            
                        # Convert tensor to PIL image format (C,H,W) -> (H,W,C)
                        masked_img_np = masked_img.squeeze().permute(1, 2, 0).cpu().numpy()
                        plt.imshow(masked_img_np)
                        plt.title(f"Masked {idx}", fontsize=8)
                        plt.axis('off')
                    
                    plt.suptitle("All Masked Images", fontsize=16)
                    plt.tight_layout()
                    plt.savefig(os.path.join(masked_images_matrix_dir, f"{base_filename}_all_masked_images_matrix.png"), dpi=150)
                    plt.close()

        # Save the final saliency maps for each detection
        for i, saliency_map in enumerate(saliency_maps[0]):
            bbox = detections[0].bounding_boxes[i].cpu().numpy()
            score = detections[0].objectness_scores[i].item()
            label = int(torch.argmax(detections[0].class_scores[i]))
            label = yolo_wrapper.model.model.names[label]  # Get class name from index
            save_saliency_images(
                image, 
                saliency_map, 
                bbox, 
                score, 
                label, 
                output_dir, 
                image_file, 
                i, 
                original_size,
                mark_high_intensity=mark_high_intensity,
                mark_high_intensity_threshold_mid=mark_high_intensity_threshold_mid,
                mark_high_intensity_threshold_high=mark_high_intensity_threshold_high
                )

        # Free memory after processing each image
        del image_tensor, detections, debug_results, saliency_maps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Also clean up the PIL image
        del image