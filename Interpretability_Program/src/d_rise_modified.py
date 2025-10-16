import os
import torch
from ultralytics import YOLO
from torchvision.transforms import ToTensor
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from vision_explanation_methods.explanations.drise import DRISE_saliency
from vision_explanation_methods.explanations.common import DetectionRecord, GeneralObjectDetectionModelWrapper
from vision_explanation_methods.explanations.drise import generate_mask, fuse_mask, compute_affinity_scores, saliency_fusion, MaskAffinityRecord
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import logging
import tempfile
import pickle
import random
logging.getLogger("ultralytics").setLevel(logging.ERROR)


def set_seed(seed: int = 42):
    """Set seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable to ensure reproducibility


def DRISE_saliency_with_debug(
        model: GeneralObjectDetectionModelWrapper,
        image_tensor: torch.Tensor,
        target_detections: List[DetectionRecord],
        number_of_masks: int,
        mask_res: Tuple[int, int] = (16, 16),
        mask_padding: Optional[int] = None,
        device: str = "cpu",
        verbose: bool = False,
        keep_all_debug_data: bool = True,
        max_debug_masks: int = 50,
) -> Dict[str, Any]:
    """Compute DRISE saliency map with memory-efficient debug information."""

    img_size = image_tensor.shape[-2:]
    if mask_padding is None:
        mask_padding = int(max(
            img_size[0] / mask_res[0], img_size[1] / mask_res[1]))

    # Create temporary directory only if debug data is needed
    temp_dir = None
    debug_indices = []
    if keep_all_debug_data:
        temp_dir = tempfile.TemporaryDirectory()
        masks_path = os.path.join(temp_dir.name, "masks")
        masked_images_path = os.path.join(temp_dir.name, "masked_images")
        predictions_path = os.path.join(temp_dir.name, "predictions")
        
        os.makedirs(masks_path, exist_ok=True)
        os.makedirs(masked_images_path, exist_ok=True)
        os.makedirs(predictions_path, exist_ok=True)
        
        # Select fewer debug indices
        debug_indices = list(range(0, number_of_masks, max(1, number_of_masks // max_debug_masks)))[:max_debug_masks]
    
    mask_records = []
    debug_mask_records = []  # Store mask records for debug indices only
    
    mask_iterator = tqdm(range(number_of_masks)) if verbose else range(number_of_masks)
    
    for i in mask_iterator:
        # Generate mask and apply to image
        mask = generate_mask(mask_res, img_size, mask_padding, device)
        masked_image = fuse_mask(image_tensor, mask)
        
        # Save debug data only for selected indices and only if requested
        if keep_all_debug_data and temp_dir and i in debug_indices:
            # Move to CPU and save immediately, then delete
            mask_cpu = mask.clone().detach().cpu()
            torch.save(mask_cpu, os.path.join(masks_path, f"mask_{i}.pt"))
            del mask_cpu
            
            masked_image_cpu = masked_image.clone().detach().cpu()
            torch.save(masked_image_cpu, os.path.join(masked_images_path, f"masked_image_{i}.pt"))
            del masked_image_cpu
        
        # Run prediction
        with torch.no_grad():
            masked_detections = model.predict(masked_image)
        
        # Save predictions only if debug data is requested
        if keep_all_debug_data and temp_dir and i in debug_indices:
            with open(os.path.join(predictions_path, f"predictions_{i}.pkl"), 'wb') as f:
                pickle.dump([det for det in masked_detections], f)
        
        # Compute affinity scores
        affinity_scores = []
        
        for (target_detection, masked_detection) in zip(target_detections, masked_detections):
            score = compute_affinity_scores(target_detection, masked_detection).detach().cpu()
            affinity_scores.append(score)
        
        # Create mask record
        mask_record = MaskAffinityRecord(
            mask=mask.clone().detach().cpu(),
            affinity_scores=affinity_scores
        )
        
        # Add to ALL mask records (needed for saliency fusion)
        mask_records.append(mask_record)
        
        # Also store in debug mask records if this is a debug index
        if keep_all_debug_data and i in debug_indices:
            debug_mask_records.append((i, mask_record))
        
        # Explicitly free GPU memory
        del mask, masked_image, masked_detections, affinity_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # Compute fused saliency maps using ALL mask records
    if verbose:
        print("Computing fused saliency maps...")
    fused_saliency_maps = saliency_fusion(mask_records, device, verbose=verbose)
    
    # Prepare minimal output
    debug_output = {
        'saliency_maps': fused_saliency_maps,
        'mask_records': debug_mask_records if keep_all_debug_data else mask_records  # Return corresponding mask records
    }
    
    # Only load debug data if explicitly requested and available
    if keep_all_debug_data and temp_dir:
        # Load debug data in smaller chunks to avoid memory spikes
        debug_output.update(_load_debug_data_efficiently(temp_dir.name, debug_indices))
        temp_dir.cleanup()
    
    return debug_output

def _load_debug_data_efficiently(temp_dir_path: str, debug_indices: List[int]) -> Dict[str, Any]:
    """Load debug data in memory-efficient chunks."""
    
    masks_path = os.path.join(temp_dir_path, "masks")
    masked_images_path = os.path.join(temp_dir_path, "masked_images")
    predictions_path = os.path.join(temp_dir_path, "predictions")
    
    # Load only essential debug data
    debug_data = {
        'masks': [],
        'masked_images': [],
        'masked_predictions': []
    }
    
    # Load in smaller batches to control memory usage
    batch_size = 10
    for i in range(0, len(debug_indices), batch_size):
        batch_indices = debug_indices[i:i+batch_size]
        
        for idx in batch_indices:
            # Load mask
            if os.path.exists(os.path.join(masks_path, f"mask_{idx}.pt")):
                mask = torch.load(os.path.join(masks_path, f"mask_{idx}.pt"), map_location='cpu')
                debug_data['masks'].append((idx, mask))
            
            # Load masked image
            if os.path.exists(os.path.join(masked_images_path, f"masked_image_{idx}.pt")):
                masked_image = torch.load(os.path.join(masked_images_path, f"masked_image_{idx}.pt"), map_location='cpu')
                debug_data['masked_images'].append((idx, masked_image))
            
            # Load predictions
            if os.path.exists(os.path.join(predictions_path, f"predictions_{idx}.pkl")):
                with open(os.path.join(predictions_path, f"predictions_{idx}.pkl"), 'rb') as f:
                    predictions = pickle.load(f)
                debug_data['masked_predictions'].append((idx, predictions))
    
    return debug_data