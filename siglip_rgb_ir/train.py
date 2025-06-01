import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import linear_sum_assignment
from datetime import datetime
import shutil
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util
from torchvision.ops import nms, batched_nms
COCO_AVAILABLE = True

# Import dataset for metadata
from datasets import FlirPairedDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train SigLIP-based object detection model")
    
    # Data arguments
    parser.add_argument(
        "--features_path",
        type=str,
        required=True,
        help="Path to saved patch embeddings pickle file (train_val combined or individual)",
    )
    parser.add_argument(
        "--ir_train_annotations",
        type=str,
        default="/arf/scratch/ebayar/datasets/FLIR_ADAS_IR/annotations/instances_train2017.json",
        help="Path to IR train annotations JSON file",
    )
    parser.add_argument(
        "--ir_val_annotations",
        type=str,
        default="/arf/scratch/ebayar/datasets/FLIR_ADAS_IR/annotations/instances_val2017.json",
        help="Path to IR validation annotations JSON file",
    )
    
    # Model arguments
    parser.add_argument(
        "--modality",
        type=str,
        choices=['rgb', 'ir', 'concat'],
        default='ir',
        help="Modality to use: 'rgb', 'ir', or 'concat' (RGB+IR concatenated)",
    )
    parser.add_argument(
        "--base_hidden_size",
        type=int,
        default=768,
        help="Base hidden size of patch embeddings (will be doubled for concat mode)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=4,  # person, bicycle, car, dog for FLIR dataset
        help="Number of object classes (background class will be added automatically)",
    )
    parser.add_argument(
        "--num_patches",
        type=int,
        default=256,  # For 384x384 with 24x24 patches = 16x16 = 256
        help="Number of patches per image",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-2,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=40,
        help="Number of warmup epochs for learning rate scheduler",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="detection_results",
        help="Base directory to save results",
    )
    
    # Training NMS arguments
    parser.add_argument(
        "--train_with_nms",
        action="store_true",
        help="Apply NMS to predictions before computing loss during training",
    )
    parser.add_argument(
        "--train_nms_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for NMS during training",
    )
    parser.add_argument(
        "--train_max_detections",
        type=int,
        default=100,
        help="Maximum number of detections per image after NMS during training",
    )
    parser.add_argument(
        "--train_confidence_threshold",
        type=float,
        default=0.1,
        help="Confidence threshold for predictions during training NMS",
    )
    
    # Loss weights
    parser.add_argument(
        "--bbox_loss_coef",
        type=float,
        default=5.0,
        help="Coefficient for bbox L1 loss",
    )
    parser.add_argument(
        "--giou_loss_coef",
        type=float,
        default=2.0,
        help="Coefficient for GIoU loss",
    )
    parser.add_argument(
        "--class_loss_coef",
        type=float,
        default=1.0,
        help="Coefficient for classification loss",
    )
    
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=5,
        help="Run COCO evaluation every N epochs",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for predictions during evaluation",
    )
    
    # NMS arguments
    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for Non-Maximum Suppression",
    )
    parser.add_argument(
        "--max_detections",
        type=int,
        default=100,
        help="Maximum number of detections per image after NMS",
    )
    parser.add_argument(
        "--enable_nms",
        action="store_true",
        help="Enable NMS during evaluation",
    )
    
    # Logging
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log every N batches",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="Save model every N epochs",
    )
    
    return parser.parse_args()


def apply_nms_per_class(boxes, scores, labels, nms_threshold=0.5, max_detections=100):
    """
    Apply Non-Maximum Suppression per class
    
    Args:
        boxes: Tensor of shape [N, 4] in xyxy format
        scores: Tensor of shape [N] with confidence scores
        labels: Tensor of shape [N] with class labels
        nms_threshold: IoU threshold for NMS
        max_detections: Maximum number of detections to keep
    
    Returns:
        Dictionary with filtered boxes, scores, and labels
    """
    if len(boxes) == 0:
        return {
            'boxes': torch.empty((0, 4), dtype=boxes.dtype, device=boxes.device),
            'scores': torch.empty(0, dtype=scores.dtype, device=scores.device),
            'labels': torch.empty(0, dtype=labels.dtype, device=labels.device)
        }
    
    # Use batched_nms which handles per-class NMS efficiently
    keep = batched_nms(boxes, scores, labels, nms_threshold)
    
    # Limit to max detections
    if len(keep) > max_detections:
        # Sort by score and keep top detections
        scores_kept = scores[keep]
        _, sorted_indices = torch.sort(scores_kept, descending=True)
        keep = keep[sorted_indices[:max_detections]]
    
    return {
        'boxes': boxes[keep],
        'scores': scores[keep],
        'labels': labels[keep],
        'indices': keep
    }


def apply_nms_to_predictions(pred_boxes, pred_logits, confidence_threshold=0.3, 
                           nms_threshold=0.5, max_detections=100, num_classes=4):
    """
    Apply NMS to model predictions
    
    Args:
        pred_boxes: Tensor of shape [num_patches, 4] in cxcywh format (normalized [0,1])
        pred_logits: Tensor of shape [num_patches, num_classes + 1]
        confidence_threshold: Minimum confidence score
        nms_threshold: IoU threshold for NMS
        max_detections: Maximum detections per image
        num_classes: Number of object classes (excluding background)
    
    Returns:
        Dictionary with final predictions after NMS
    """
    device = pred_boxes.device
    
    # Convert logits to probabilities
    pred_probs = torch.softmax(pred_logits, dim=-1)
    
    # Get predictions for non-background classes only
    # Background class is at index num_classes
    object_probs = pred_probs[:, :num_classes]  # [num_patches, num_classes]
    
    # Find all predictions above confidence threshold
    max_probs, predicted_classes = torch.max(object_probs, dim=1)
    valid_mask = max_probs > confidence_threshold
    
    if not valid_mask.any():
        # No valid detections
        return {
            'boxes': torch.empty((0, 4), dtype=pred_boxes.dtype, device=device),
            'scores': torch.empty(0, dtype=pred_boxes.dtype, device=device),
            'labels': torch.empty(0, dtype=torch.long, device=device)
        }
    
    # Filter valid predictions
    valid_boxes = pred_boxes[valid_mask]  # cxcywh format
    valid_scores = max_probs[valid_mask]
    valid_labels = predicted_classes[valid_mask]
    
    # Convert boxes from cxcywh to xyxy format for NMS
    valid_boxes_xyxy = box_cxcywh_to_xyxy(valid_boxes)
    
    # Apply NMS
    nms_result = apply_nms_per_class(
        valid_boxes_xyxy, valid_scores, valid_labels, 
        nms_threshold, max_detections
    )
    
    # Convert boxes back to cxcywh format if needed
    if len(nms_result['boxes']) > 0:
        nms_result['boxes_cxcywh'] = box_xyxy_to_cxcywh(nms_result['boxes'])
    else:
        nms_result['boxes_cxcywh'] = torch.empty((0, 4), dtype=pred_boxes.dtype, device=device)
    
    return {
        'boxes': nms_result['boxes'],  # xyxy format
        'boxes_cxcywh': nms_result['boxes_cxcywh'],  # cxcywh format
        'scores': nms_result['scores'],
        'labels': nms_result['labels']
    }


def apply_nms_for_training(pred_boxes, pred_logits, confidence_threshold=0.1, 
                          nms_threshold=0.5, max_detections=100, num_classes=4):
    """
    Apply NMS to predictions for training and return indices of kept predictions
    This allows backpropagation through the original predictions
    
    Args:
        pred_boxes: Tensor of shape [num_patches, 4] in cxcywh format (normalized [0,1])
        pred_logits: Tensor of shape [num_patches, num_classes + 1]
        confidence_threshold: Minimum confidence score
        nms_threshold: IoU threshold for NMS
        max_detections: Maximum detections per image
        num_classes: Number of object classes (excluding background)
    
    Returns:
        Dictionary with filtered predictions and their original indices
    """
    device = pred_boxes.device
    
    # Convert logits to probabilities
    pred_probs = torch.softmax(pred_logits, dim=-1)
    
    # Get predictions for non-background classes only
    object_probs = pred_probs[:, :num_classes]  # [num_patches, num_classes]
    
    # Find all predictions above confidence threshold
    max_probs, predicted_classes = torch.max(object_probs, dim=1)
    valid_mask = max_probs > confidence_threshold
    
    if not valid_mask.any():
        # No valid detections - return empty tensors with proper indices
        return {
            'filtered_boxes': torch.empty((0, 4), dtype=pred_boxes.dtype, device=device),
            'filtered_logits': torch.empty((0, num_classes + 1), dtype=pred_logits.dtype, device=device),
            'original_indices': torch.empty(0, dtype=torch.long, device=device),
            'valid_mask': valid_mask,
            'num_filtered': 0  # FIX: Always include num_filtered
        }
    
    # Get indices of valid predictions
    valid_indices = torch.where(valid_mask)[0]
    
    # Filter valid predictions
    valid_boxes = pred_boxes[valid_mask]  # cxcywh format
    valid_logits = pred_logits[valid_mask]
    valid_scores = max_probs[valid_mask]
    valid_labels = predicted_classes[valid_mask]
    
    # Convert boxes from cxcywh to xyxy format for NMS
    valid_boxes_xyxy = box_cxcywh_to_xyxy(valid_boxes)
    
    # Apply NMS
    nms_result = apply_nms_per_class(
        valid_boxes_xyxy, valid_scores, valid_labels, 
        nms_threshold, max_detections
    )
    
    if len(nms_result['indices']) == 0:
        # NMS filtered out all detections
        return {
            'filtered_boxes': torch.empty((0, 4), dtype=pred_boxes.dtype, device=device),
            'filtered_logits': torch.empty((0, num_classes + 1), dtype=pred_logits.dtype, device=device),
            'original_indices': torch.empty(0, dtype=torch.long, device=device),
            'valid_mask': valid_mask,
            'num_filtered': 0  # FIX: Always include num_filtered
        }
    
    # Get the original indices of the NMS-kept predictions
    nms_kept_indices = nms_result['indices']  # indices within valid predictions
    original_kept_indices = valid_indices[nms_kept_indices]  # indices in original prediction tensor
    
    # Get the filtered predictions in original cxcywh format
    filtered_boxes = pred_boxes[original_kept_indices]
    filtered_logits = pred_logits[original_kept_indices]
    
    return {
        'filtered_boxes': filtered_boxes,
        'filtered_logits': filtered_logits,
        'original_indices': original_kept_indices,
        'valid_mask': valid_mask,
        'num_filtered': len(original_kept_indices)
    }


def filter_predictions_for_training(outputs, confidence_threshold=0.1, nms_threshold=0.5, 
                                  max_detections=100, num_classes=4):
    """
    Filter batch predictions for training with NMS
    
    Args:
        outputs: Model outputs dict with 'pred_boxes' and 'pred_logits'
        confidence_threshold: Minimum confidence score
        nms_threshold: IoU threshold for NMS
        max_detections: Maximum detections per image
        num_classes: Number of object classes
    
    Returns:
        Dictionary with filtered outputs that can be used for loss computation
    """
    batch_size = outputs['pred_boxes'].shape[0]
    
    filtered_outputs = {
        'pred_boxes': [],
        'pred_logits': [],
        'original_indices': [],  # Track which original predictions were kept
        'batch_info': []  # Info about filtering for each sample
    }
    
    total_original = 0
    total_filtered = 0
    
    for i in range(batch_size):
        pred_boxes = outputs['pred_boxes'][i]  # [num_patches, 4]
        pred_logits = outputs['pred_logits'][i]  # [num_patches, num_classes + 1]
        
        # Apply NMS filtering
        nms_result = apply_nms_for_training(
            pred_boxes, pred_logits, confidence_threshold,
            nms_threshold, max_detections, num_classes
        )
        
        filtered_outputs['pred_boxes'].append(nms_result['filtered_boxes'])
        filtered_outputs['pred_logits'].append(nms_result['filtered_logits'])
        filtered_outputs['original_indices'].append(nms_result['original_indices'])
        
        # Track statistics
        num_original = pred_boxes.shape[0]
        num_filtered = nms_result['num_filtered']
        total_original += num_original
        total_filtered += num_filtered
        
        filtered_outputs['batch_info'].append({
            'sample_idx': i,
            'original_count': num_original,
            'filtered_count': num_filtered,
            'filter_ratio': num_filtered / num_original if num_original > 0 else 0
        })
    
    # Convert lists to tensors for loss computation
    if total_filtered > 0:
        filtered_outputs['pred_boxes'] = torch.cat(filtered_outputs['pred_boxes'], dim=0)
        filtered_outputs['pred_logits'] = torch.cat(filtered_outputs['pred_logits'], dim=0)
        filtered_outputs['original_indices'] = torch.cat(filtered_outputs['original_indices'], dim=0)
    else:
        device = outputs['pred_boxes'].device
        filtered_outputs['pred_boxes'] = torch.empty((0, 4), device=device)
        filtered_outputs['pred_logits'] = torch.empty((0, outputs['pred_logits'].shape[-1]), device=device)
        filtered_outputs['original_indices'] = torch.empty(0, dtype=torch.long, device=device)
    
    filtered_outputs['statistics'] = {
        'total_original': total_original,
        'total_filtered': total_filtered,
        'overall_filter_ratio': total_filtered / total_original if total_original > 0 else 0
    }
    
    return filtered_outputs


def postprocess_predictions(outputs, confidence_threshold=0.3, nms_threshold=0.5, 
                          max_detections=100, num_classes=4, enable_nms=True):
    """
    Post-process model outputs to get final predictions
    
    Args:
        outputs: Model outputs dict with 'pred_boxes' and 'pred_logits'
        confidence_threshold: Minimum confidence score
        nms_threshold: IoU threshold for NMS
        max_detections: Maximum detections per image
        num_classes: Number of object classes
        enable_nms: Whether to apply NMS
    
    Returns:
        List of prediction dictionaries for each image in batch
    """
    batch_size = outputs['pred_boxes'].shape[0]
    batch_predictions = []
    
    for i in range(batch_size):
        pred_boxes = outputs['pred_boxes'][i]  # [num_patches, 4]
        pred_logits = outputs['pred_logits'][i]  # [num_patches, num_classes + 1]
        
        if enable_nms:
            # Apply NMS
            predictions = apply_nms_to_predictions(
                pred_boxes, pred_logits, confidence_threshold,
                nms_threshold, max_detections, num_classes
            )
        else:
            # Simple thresholding without NMS
            pred_probs = torch.softmax(pred_logits, dim=-1)
            object_probs = pred_probs[:, :num_classes]
            max_probs, predicted_classes = torch.max(object_probs, dim=1)
            valid_mask = max_probs > confidence_threshold
            
            predictions = {
                'boxes': box_cxcywh_to_xyxy(pred_boxes[valid_mask]),
                'boxes_cxcywh': pred_boxes[valid_mask],
                'scores': max_probs[valid_mask],
                'labels': predicted_classes[valid_mask]
            }
        
        batch_predictions.append(predictions)
    
    return batch_predictions


def create_output_directory(args):
    """Create output directory with timestamp and parameters"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create descriptive folder name
    nms_suffix = f"_nms{args.nms_threshold}" if args.enable_nms else "_no_nms"
    train_nms_suffix = f"_train_nms{args.train_nms_threshold}" if args.train_with_nms else "_no_train_nms"
    
    dir_name = (f"{args.modality}_lr{args.learning_rate}_bs{args.batch_size}_"
                f"epochs{args.num_epochs}_bbox{args.bbox_loss_coef}_"
                f"giou{args.giou_loss_coef}_class{args.class_loss_coef}"
                f"{train_nms_suffix}{nms_suffix}_{timestamp}")
    
    output_dir = os.path.join(args.output_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['models', 'plots', 'logs', 'metrics', 'evaluation']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # Save arguments
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Created output directory: {output_dir}")
    return output_dir


class MetricsLogger:
    """Logger for training and validation metrics"""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'train_bbox_loss': [],
            'train_giou_loss': [],
            'train_ce_loss': [],
            'val_loss': [],
            'val_bbox_loss': [],
            'val_giou_loss': [],
            'val_ce_loss': [],
            'learning_rate': [],
            'ap': [],
            'ap_50': [],
            'ap_75': [],
            'ap_small': [],
            'ap_medium': [],
            'ap_large': [],
            'ar_1': [],
            'ar_10': [],
            'ar_100': [],
            'ar_small': [],
            'ar_medium': [],
            'ar_large': []
        }
    
    def log_epoch(self, epoch, train_metrics, val_metrics, lr, ap_metrics=None):
        """Log metrics for an epoch"""
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_metrics['total_loss'])
        self.metrics['train_bbox_loss'].append(train_metrics['bbox_loss'])
        self.metrics['train_giou_loss'].append(train_metrics['giou_loss'])
        self.metrics['train_ce_loss'].append(train_metrics['ce_loss'])
        self.metrics['val_loss'].append(val_metrics['total_loss'])
        self.metrics['val_bbox_loss'].append(val_metrics['bbox_loss'])
        self.metrics['val_giou_loss'].append(val_metrics['giou_loss'])
        self.metrics['val_ce_loss'].append(val_metrics['ce_loss'])
        self.metrics['learning_rate'].append(lr)
        
        if ap_metrics:
            self.metrics['ap'].append(ap_metrics.get('AP', 0))
            self.metrics['ap_50'].append(ap_metrics.get('AP_50', 0))
            self.metrics['ap_75'].append(ap_metrics.get('AP_75', 0))
            self.metrics['ap_small'].append(ap_metrics.get('AP_small', 0))
            self.metrics['ap_medium'].append(ap_metrics.get('AP_medium', 0))
            self.metrics['ap_large'].append(ap_metrics.get('AP_large', 0))
            self.metrics['ar_1'].append(ap_metrics.get('AR_1', 0))
            self.metrics['ar_10'].append(ap_metrics.get('AR_10', 0))
            self.metrics['ar_100'].append(ap_metrics.get('AR_100', 0))
            self.metrics['ar_small'].append(ap_metrics.get('AR_small', 0))
            self.metrics['ar_medium'].append(ap_metrics.get('AR_medium', 0))
            self.metrics['ar_large'].append(ap_metrics.get('AR_large', 0))
        else:
            # Fill with None/0 if no AP metrics
            for key in ['ap', 'ap_50', 'ap_75', 'ap_small', 'ap_medium', 'ap_large',
                       'ar_1', 'ar_10', 'ar_100', 'ar_small', 'ar_medium', 'ar_large']:
                self.metrics[key].append(None)
    
    def save_metrics(self):
        """Save metrics to CSV"""
        df = pd.DataFrame(self.metrics)
        csv_path = os.path.join(self.output_dir, 'metrics', 'training_metrics.csv')
        df.to_csv(csv_path, index=False)
        
        # Also save as JSON
        json_path = os.path.join(self.output_dir, 'metrics', 'training_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Metrics saved to {csv_path} and {json_path}")
    
    def plot_metrics(self):
        """Create and save plots"""
        epochs = self.metrics['epoch']
        
        # Plot training and validation losses
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Total loss
        axes[0, 0].plot(epochs, self.metrics['train_loss'], label='Train', marker='o')
        axes[0, 0].plot(epochs, self.metrics['val_loss'], label='Validation', marker='s')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Component losses
        axes[0, 1].plot(epochs, self.metrics['train_bbox_loss'], label='Train BBox', marker='o')
        axes[0, 1].plot(epochs, self.metrics['val_bbox_loss'], label='Val BBox', marker='s')
        axes[0, 1].plot(epochs, self.metrics['train_giou_loss'], label='Train GIoU', marker='^')
        axes[0, 1].plot(epochs, self.metrics['val_giou_loss'], label='Val GIoU', marker='v')
        axes[0, 1].set_title('BBox and GIoU Losses')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Classification loss
        axes[1, 0].plot(epochs, self.metrics['train_ce_loss'], label='Train CE', marker='o')
        axes[1, 0].plot(epochs, self.metrics['val_ce_loss'], label='Val CE', marker='s')
        axes[1, 0].set_title('Classification Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(epochs, self.metrics['learning_rate'], label='Learning Rate', marker='o')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'training_losses.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot AP/AR metrics if available
        ap_epochs = [i for i, ap in enumerate(self.metrics['ap']) if ap is not None]
        if ap_epochs:
            ap_values = [self.metrics['ap'][i] for i in ap_epochs]
            ap_50_values = [self.metrics['ap_50'][i] for i in ap_epochs]
            ap_75_values = [self.metrics['ap_75'][i] for i in ap_epochs]
            ar_values = [self.metrics['ar_100'][i] for i in ap_epochs]
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # AP metrics
            axes[0].plot([epochs[i] for i in ap_epochs], ap_values, label='AP', marker='o')
            axes[0].plot([epochs[i] for i in ap_epochs], ap_50_values, label='AP@0.5', marker='s')
            axes[0].plot([epochs[i] for i in ap_epochs], ap_75_values, label='AP@0.75', marker='^')
            axes[0].set_title('Average Precision (AP)')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('AP')
            axes[0].legend()
            axes[0].grid(True)
            
            # AR metrics
            ar_1_values = [self.metrics['ar_1'][i] for i in ap_epochs]
            ar_10_values = [self.metrics['ar_10'][i] for i in ap_epochs]
            ar_100_values = [self.metrics['ar_100'][i] for i in ap_epochs]
            
            axes[1].plot([epochs[i] for i in ap_epochs], ar_1_values, label='AR@1', marker='o')
            axes[1].plot([epochs[i] for i in ap_epochs], ar_10_values, label='AR@10', marker='s')
            axes[1].plot([epochs[i] for i in ap_epochs], ar_100_values, label='AR@100', marker='^')
            axes[1].set_title('Average Recall (AR)')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('AR')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'plots', 'ap_ar_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()


# Box operations (provided by user)
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    from torchvision.ops.boxes import box_area
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


class BoxPredictionHead(nn.Module):
    """Bounding box prediction head similar to OWLv2"""
    def __init__(self, hidden_size: int, out_dim: int = 4):
        super().__init__()
        self.dense0 = nn.Linear(hidden_size, hidden_size)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(hidden_size, out_dim)

    def forward(self, patch_features: torch.Tensor) -> torch.FloatTensor:
        # patch_features: [batch_size, num_patches, hidden_size]
        output = self.dense0(patch_features)
        output = self.gelu(output)
        output = self.dense1(output)
        output = self.gelu(output)
        output = self.dense2(output)  # [batch_size, num_patches, 4]
        return torch.sigmoid(output)  # Normalize to [0, 1]


class ClassPredictionHead(nn.Module):
    """Classification prediction head similar to OWLv2"""
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.dense0 = nn.Linear(hidden_size, hidden_size)
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()
        self.classifier = nn.Linear(hidden_size, num_classes + 1)  # +1 for background

    def forward(self, patch_features: torch.Tensor) -> torch.FloatTensor:
        # patch_features: [batch_size, num_patches, hidden_size]
        output = self.dense0(patch_features)
        output = self.gelu(output)
        output = self.dense1(output)
        output = self.gelu(output)
        output = self.classifier(output)  # [batch_size, num_patches, num_classes + 1]
        return output


class SigLIPDetectionModel(nn.Module):
    """SigLIP-based object detection model"""
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Detection heads
        self.box_head = BoxPredictionHead(hidden_size)
        self.class_head = ClassPredictionHead(hidden_size, num_classes)
        
        # Layer norm (similar to OWLv2)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, patch_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            patch_embeddings: [batch_size, num_patches, hidden_size]
        
        Returns:
            Dict with 'pred_boxes' and 'pred_logits'
        """
        # Apply layer norm
        normalized_features = self.layer_norm(patch_embeddings)
        
        # Predictions
        pred_boxes = self.box_head(normalized_features)  # [B, N, 4]
        pred_logits = self.class_head(normalized_features)  # [B, N, num_classes + 1]
        
        return {
            'pred_boxes': pred_boxes,
            'pred_logits': pred_logits
        }


class PatchEmbeddingDataset(Dataset):
    """Dataset for pre-computed patch embeddings with ground truth"""
    def __init__(self, features_data: Dict, modality: str = 'ir'):
        # features_data should be the data for a specific split (train or val)
        self.rgb_patch_embeddings = features_data['rgb_patch_embeddings']
        self.ir_patch_embeddings = features_data['ir_patch_embeddings']
        self.metadata = features_data['metadata']
        self.modality = modality
        self.split = features_data.get('split', 'unknown')
        
        # Validate modality
        if modality not in ['rgb', 'ir', 'concat']:
            raise ValueError(f"Invalid modality: {modality}. Must be 'rgb', 'ir', or 'concat'")
        
        # Create category mapping
        self.category_mapping = self._create_category_mapping()
        
        print(f"Dataset created for {modality} modality ({self.split} split) with {len(self.metadata)} samples")
        if modality == 'concat':
            print(f"Concatenated embeddings will have shape: [num_patches, {self.rgb_patch_embeddings.shape[-1] * 2}]")
        
    def _create_category_mapping(self):
        """Create mapping from category names to indices"""
        # FLIR dataset has 4 main categories: person, bicycle, car, dog
        categories = ['person', 'bicycle', 'car', 'dog']
        return {name: idx for idx, name in enumerate(categories)}
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get patch embeddings based on modality
        if self.modality == 'rgb':
            patches = torch.from_numpy(self.rgb_patch_embeddings[idx]).float()  # [num_patches, hidden_size]
        elif self.modality == 'ir':
            patches = torch.from_numpy(self.ir_patch_embeddings[idx]).float()   # [num_patches, hidden_size]
        elif self.modality == 'concat':
            rgb_patches = torch.from_numpy(self.rgb_patch_embeddings[idx]).float()  # [num_patches, hidden_size]
            ir_patches = torch.from_numpy(self.ir_patch_embeddings[idx]).float()    # [num_patches, hidden_size]
            patches = torch.cat([rgb_patches, ir_patches], dim=-1)  # [num_patches, hidden_size * 2]
        
        # Get metadata
        meta = self.metadata[idx]
        
        # Prepare targets
        targets = self._prepare_targets(meta)
        
        return {
            'patches': patches,
            'targets': targets,
            'image_id': meta['image_id'],
            'modality': self.modality
        }
    
    def _prepare_targets(self, meta: Dict) -> Dict:
        """Prepare ground truth targets"""
        boxes = meta['boxes']
        class_names = meta['class_names']
        img_size = (512, 640)
        
        if len(boxes) == 0:
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.long),
            }
        
        # Convert boxes to tensor - boxes are in xyxy pixel coordinates
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        
        # Debug: Print original boxes for first few samples
        if not hasattr(self, '_target_debug_printed'):
            print(f"\nTarget preparation debug:")
            print(f"  Image size (H, W): {img_size}")
            print(f"  Original boxes (pixel xyxy): {boxes_tensor[:3] if len(boxes_tensor) >= 3 else boxes_tensor}")
            self._target_debug_printed = True
        
        # Normalize boxes to [0, 1] using image dimensions
        if boxes_tensor.numel() > 0:
            img_height, img_width = img_size
            
            # Normalize: [x1, y1, x2, y2] in pixels -> [x1/w, y1/h, x2/w, y2/h] in [0,1]
            boxes_tensor[:, [0, 2]] /= img_width   # x coordinates
            boxes_tensor[:, [1, 3]] /= img_height  # y coordinates
            
            # Debug: Print normalized boxes
            if not hasattr(self, '_norm_debug_printed'):
                print(f"  Normalized boxes (xyxy): {boxes_tensor[:3] if len(boxes_tensor) >= 3 else boxes_tensor}")
                self._norm_debug_printed = True
            
            # Ensure boxes are properly bounded [0, 1] after normalization
            boxes_tensor = torch.clamp(boxes_tensor, 0, 1)
            
            # Convert from xyxy to cxcywh format (expected by loss functions)
            boxes_tensor = box_xyxy_to_cxcywh(boxes_tensor)
            
            # Debug: Print final boxes
            if not hasattr(self, '_final_debug_printed'):
                print(f"  Final boxes (cxcywh): {boxes_tensor[:3] if len(boxes_tensor) >= 3 else boxes_tensor}")
                self._final_debug_printed = True
        
        # Convert class names to indices
        labels = []
        for class_name in class_names:
            if class_name in self.category_mapping:
                labels.append(self.category_mapping[class_name])
            else:
                print(f"Warning: Unknown class name '{class_name}', mapping to class 0")
                labels.append(0)  # Default to first class if unknown
        
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
        }


class HungarianMatcher(nn.Module):
    """Hungarian matcher for assignment of predictions to targets"""
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets, filtered_info=None):
        """
        Args:
            outputs: Dict with 'pred_logits' [B, N, num_classes] and 'pred_boxes' [B, N, 4]
                     OR filtered outputs with different structure
            targets: List of dicts with 'labels' and 'boxes'
            filtered_info: Optional info about NMS filtering for training
        
        Returns:
            List of (pred_idx, target_idx) for each sample in batch
        """
        if filtered_info is not None:
            # Handle NMS-filtered predictions for training
            return self._forward_filtered(outputs, targets, filtered_info)
        else:
            # Standard matching for unfiltered predictions
            return self._forward_standard(outputs, targets)
    
    def _forward_standard(self, outputs, targets):
        """Standard Hungarian matching for unfiltered predictions"""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Flatten to compute costs
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # Concat all target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # Classification cost
        cost_class = -out_prob[:, tgt_ids]
        
        # L1 cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # GIoU cost
        if len(tgt_bbox) > 0 and len(out_bbox) > 0:
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_bbox)
            )
        else:
            cost_giou = torch.zeros_like(cost_bbox)
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
    def _forward_filtered(self, filtered_outputs, targets, filtered_info):
        """Hungarian matching for NMS-filtered predictions"""
        if len(filtered_outputs['pred_boxes']) == 0:
            # No predictions after filtering
            return [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)) for _ in targets]
        
        # Get filtered predictions
        out_prob = filtered_outputs['pred_logits'].softmax(-1)  # [total_filtered_preds, num_classes]
        out_bbox = filtered_outputs['pred_boxes']  # [total_filtered_preds, 4]
        
        # Concat all target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        if len(tgt_ids) == 0:
            # No targets
            return [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)) for _ in targets]
        
        # Classification cost
        cost_class = -out_prob[:, tgt_ids]
        
        # L1 cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # GIoU cost
        if len(tgt_bbox) > 0 and len(out_bbox) > 0:
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_bbox)
            )
        else:
            cost_giou = torch.zeros_like(cost_bbox)
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.cpu()
        
        # Split cost matrix by batch samples based on filtered_info
        batch_sizes = []
        current_idx = 0
        
        for batch_info in filtered_info['batch_info']:
            filtered_count = batch_info['filtered_count']
            batch_sizes.append(filtered_count)
            current_idx += filtered_count
        
        # Split cost matrix by batches and targets
        target_sizes = [len(v["boxes"]) for v in targets]
        cost_splits = []
        pred_start = 0
        
        for pred_count in batch_sizes:
            if pred_count > 0:
                cost_splits.append(C[pred_start:pred_start + pred_count])
            else:
                # No predictions for this batch item
                cost_splits.append(torch.empty(0, len(tgt_ids), device='cpu'))
            pred_start += pred_count
        
        # Apply Hungarian algorithm per batch item
        indices = []
        target_start = 0
        
        for i, (cost_matrix, target_count) in enumerate(zip(cost_splits, target_sizes)):
            if cost_matrix.shape[0] > 0 and target_count > 0:
                # Extract cost matrix for this batch item's targets
                target_end = target_start + target_count
                batch_cost = cost_matrix[:, target_start:target_end]
                
                # Apply Hungarian algorithm
                pred_idx, target_idx = linear_sum_assignment(batch_cost)
                
                # Convert to tensors and add batch offset for predictions
                pred_idx = torch.as_tensor(pred_idx, dtype=torch.int64)
                target_idx = torch.as_tensor(target_idx, dtype=torch.int64)
                
                # Map filtered prediction indices back to batch-relative indices
                if len(pred_idx) > 0:
                    batch_start = sum(batch_sizes[:i])
                    pred_idx = pred_idx + batch_start
                
                indices.append((pred_idx, target_idx))
            else:
                # No valid matches for this batch item
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
            
            target_start += target_count
        
        return indices


class DetectionLoss(nn.Module):
    """Detection loss computation with support for NMS-filtered predictions"""
    def __init__(self, num_classes: int, matcher: HungarianMatcher, 
                 bbox_loss_coef: float, giou_loss_coef: float, class_loss_coef: float):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        self.class_loss_coef = class_loss_coef

    def _get_src_permutation_idx(self, indices):
        # Permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_src_permutation_idx_filtered(self, indices, filtered_info):
        """Get permutation indices for filtered predictions"""
        # indices are already relative to the filtered predictions
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, num_boxes, filtered_info=None, original_outputs=None):
        """Classification loss with support for filtered predictions"""
        if filtered_info is not None:
            return self._loss_labels_filtered(outputs, targets, indices, num_boxes, filtered_info, original_outputs)
        else:
            return self._loss_labels_standard(outputs, targets, indices, num_boxes)
    
    def _loss_labels_standard(self, outputs, targets, indices, num_boxes):
        """Standard classification loss for unfiltered predictions"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # Background class has index = num_classes
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                   dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        losses = {'loss_ce': loss_ce}
        
        return losses
    
    def _loss_labels_filtered(self, filtered_outputs, targets, indices, num_boxes, filtered_info, original_outputs):
        """Classification loss for NMS-filtered predictions"""
        if len(filtered_outputs['pred_logits']) == 0:
            # No filtered predictions
            device = original_outputs['pred_logits'].device
            return {'loss_ce': torch.tensor(0.0, device=device, requires_grad=True)}
        
        # Get filtered predictions and their original indices
        filtered_logits = filtered_outputs['pred_logits']  # [total_filtered, num_classes+1]
        original_indices = filtered_info['original_indices']  # [total_filtered]
        
        # Build target classes for filtered predictions
        idx = self._get_src_permutation_idx_filtered(indices, filtered_info)
        
        if len(idx[0]) > 0:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            
            # Create target tensor for filtered predictions
            target_classes = torch.full((len(filtered_logits),), self.num_classes,
                                       dtype=torch.int64, device=filtered_logits.device)
            target_classes[idx[1]] = target_classes_o
            
            # Compute loss on filtered predictions
            loss_ce = F.cross_entropy(filtered_logits, target_classes)
        else:
            # No matches, all predictions are background
            target_classes = torch.full((len(filtered_logits),), self.num_classes,
                                       dtype=torch.int64, device=filtered_logits.device)
            loss_ce = F.cross_entropy(filtered_logits, target_classes)
        
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, filtered_info=None, original_outputs=None):
        """Bounding box loss with support for filtered predictions"""
        if filtered_info is not None:
            return self._loss_boxes_filtered(outputs, targets, indices, num_boxes, filtered_info, original_outputs)
        else:
            return self._loss_boxes_standard(outputs, targets, indices, num_boxes)
    
    def _loss_boxes_standard(self, outputs, targets, indices, num_boxes):
        """Standard bounding box loss for unfiltered predictions"""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if len(src_boxes) == 0:
            device = outputs['pred_boxes'].device
            losses = {
                'loss_bbox': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_giou': torch.tensor(0.0, device=device, requires_grad=True)
            }
            return losses

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses
    
    def _loss_boxes_filtered(self, filtered_outputs, targets, indices, num_boxes, filtered_info, original_outputs):
        """Bounding box loss for NMS-filtered predictions"""
        if len(filtered_outputs['pred_boxes']) == 0:
            device = original_outputs['pred_boxes'].device
            losses = {
                'loss_bbox': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_giou': torch.tensor(0.0, device=device, requires_grad=True)
            }
            return losses
        
        # Get filtered boxes
        filtered_boxes = filtered_outputs['pred_boxes']  # [total_filtered, 4]
        
        # Get matched boxes and targets
        idx = self._get_src_permutation_idx_filtered(indices, filtered_info)
        
        if len(idx[0]) == 0:
            # No matches
            device = filtered_boxes.device
            losses = {
                'loss_bbox': torch.tensor(0.0, device=device, requires_grad=True),
                'loss_giou': torch.tensor(0.0, device=device, requires_grad=True)
            }
            return losses
        
        src_boxes = filtered_boxes[idx[1]]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def forward(self, outputs, targets, filtered_info=None, original_outputs=None):
        """
        Compute the losses
        
        Args:
            outputs: Model outputs (filtered or unfiltered)
            targets: Ground truth targets
            filtered_info: Info about NMS filtering (None for standard mode)
            original_outputs: Original unfiltered outputs (for filtered mode)
        """
        # Retrieve the matching between the outputs and the targets
        indices = self.matcher(outputs, targets, filtered_info)

        # Compute the average number of target boxes across all nodes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, 
                                   device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        
        # Classification loss
        loss_dict = self.loss_labels(outputs, targets, indices, num_boxes, filtered_info, original_outputs)
        losses.update(loss_dict)
        
        # Box losses
        loss_dict = self.loss_boxes(outputs, targets, indices, num_boxes, filtered_info, original_outputs)
        losses.update(loss_dict)

        # Total loss
        total_loss = (self.class_loss_coef * losses['loss_ce'] + 
                     self.bbox_loss_coef * losses['loss_bbox'] + 
                     self.giou_loss_coef * losses['loss_giou'])
        
        losses['total_loss'] = total_loss
        
        # Add filtering statistics if available
        if filtered_info is not None:
            losses['filter_stats'] = filtered_info['statistics']
        
        return losses


class COCOEvaluator:
    """COCO-style evaluation for object detection with NMS support"""
    def __init__(self, output_dir, category_mapping, enable_nms=True, nms_threshold=0.5, max_detections=100):
        self.output_dir = output_dir
        self.category_mapping = category_mapping
        self.inv_category_mapping = {v: k for k, v in category_mapping.items()}
        self.enable_nms = enable_nms
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        
    def convert_to_coco_format(self, predictions, targets, image_ids):
        """Convert predictions and targets to COCO format"""
        coco_predictions = []
        coco_targets = []
        
        pred_id = 1
        for i, (pred, target, img_id) in enumerate(zip(predictions, targets, image_ids)):
            # Predictions are already post-processed (with or without NMS)
            pred_boxes = pred['boxes']  # xyxy format, normalized [0,1]
            pred_scores = pred['scores']
            pred_labels = pred['labels']
            
            # Convert predictions to COCO format
            for j in range(len(pred_boxes)):
                box = pred_boxes[j]  # xyxy format, normalized
                score = pred_scores[j].item()
                label = pred_labels[j].item()
                
                # Convert normalized coordinates to pixel coordinates
                img_w, img_h = 640, 512
                x1, y1, x2, y2 = box
                x1_pix = x1 * img_w
                y1_pix = y1 * img_h
                x2_pix = x2 * img_w
                y2_pix = y2 * img_h
                
                coco_predictions.append({
                    'id': pred_id,
                    'image_id': int(img_id),
                    'category_id': label + 1,  # COCO categories start from 1
                    'bbox': [x1_pix.item(), y1_pix.item(), (x2_pix-x1_pix).item(), (y2_pix-y1_pix).item()],  # [x, y, w, h]
                    'score': score,
                    'area': ((x2_pix-x1_pix) * (y2_pix-y1_pix)).item()
                })
                pred_id += 1
            
            # Convert targets
            target_boxes = target['boxes'].cpu()  # [num_objects, 4] in cxcywh format
            target_labels = target['labels'].cpu()  # [num_objects]
            
            for obj_idx in range(target_boxes.shape[0]):
                box = target_boxes[obj_idx]  # cxcywh format
                label = target_labels[obj_idx].item()
                
                # Convert to xyxy format and scale to image size
                img_w, img_h = 640, 512
                x_center, y_center, width, height = box
                x1 = (x_center - width / 2) * img_w
                y1 = (y_center - height / 2) * img_h
                x2 = (x_center + width / 2) * img_w
                y2 = (y_center + height / 2) * img_h
                
                coco_targets.append({
                    'id': len(coco_targets) + 1,
                    'image_id': int(img_id),
                    'category_id': label + 1,  # COCO categories start from 1
                    'bbox': [x1.item(), y1.item(), (x2-x1).item(), (y2-y1).item()],  # [x, y, w, h]
                    'area': ((x2-x1) * (y2-y1)).item(),
                    'iscrowd': 0
                })
        
        return coco_predictions, coco_targets
    
    def evaluate(self, model, dataloader, device, confidence_threshold=0.5):
        """Run COCO evaluation with NMS support"""
        if not COCO_AVAILABLE:
            print("pycocotools not available, skipping COCO evaluation")
            return {}
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_image_ids = []
        
        print(f"Running evaluation with NMS {'enabled' if self.enable_nms else 'disabled'}")
        if self.enable_nms:
            print(f"NMS threshold: {self.nms_threshold}, Max detections: {self.max_detections}")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Running evaluation"):
                patches = batch['patches'].to(device)
                targets = batch['targets']
                image_ids = batch['image_ids']
                
                # Forward pass
                outputs = model(patches)
                
                # Post-process predictions with or without NMS
                batch_predictions = postprocess_predictions(
                    outputs, 
                    confidence_threshold=confidence_threshold,
                    nms_threshold=self.nms_threshold,
                    max_detections=self.max_detections,
                    num_classes=len(self.category_mapping),
                    enable_nms=self.enable_nms
                )
                
                # Collect predictions and targets
                for i in range(len(targets)):
                    all_predictions.append(batch_predictions[i])
                    all_targets.append(targets[i])
                    all_image_ids.append(image_ids[i])
        
        # Convert to COCO format
        coco_predictions, coco_targets = self.convert_to_coco_format(
            all_predictions, all_targets, all_image_ids
        )
        
        print(f"Generated {len(coco_predictions)} predictions and {len(coco_targets)} targets")
        
        if len(coco_predictions) == 0 or len(coco_targets) == 0:
            print("No predictions or targets found for evaluation")
            return {}
        
        # Create COCO dataset structure
        categories = [
            {'id': i + 1, 'name': name} 
            for name, i in self.category_mapping.items()
        ]
        
        images = [
            {'id': int(img_id)} 
            for img_id in set(all_image_ids)
        ]
        
        coco_gt_dict = {
            'images': images,
            'annotations': coco_targets,
            'categories': categories
        }
        
        # Save COCO format files
        nms_suffix = f"_nms{self.nms_threshold}" if self.enable_nms else "_no_nms"
        gt_path = os.path.join(self.output_dir, 'evaluation', f'coco_gt{nms_suffix}.json')
        pred_path = os.path.join(self.output_dir, 'evaluation', f'coco_predictions{nms_suffix}.json')
        
        with open(gt_path, 'w') as f:
            json.dump(coco_gt_dict, f)
        
        with open(pred_path, 'w') as f:
            json.dump(coco_predictions, f)
        
        try:
            # Load COCO ground truth
            coco_gt = COCO(gt_path)
            
            # Load predictions
            coco_dt = coco_gt.loadRes(pred_path)
            
            # Run evaluation
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract metrics
            metrics = {
                'AP': coco_eval.stats[0],      # AP @ IoU=0.50:0.95
                'AP_50': coco_eval.stats[1],   # AP @ IoU=0.50
                'AP_75': coco_eval.stats[2],   # AP @ IoU=0.75
                'AP_small': coco_eval.stats[3],    # AP for small objects
                'AP_medium': coco_eval.stats[4],   # AP for medium objects
                'AP_large': coco_eval.stats[5],    # AP for large objects
                'AR_1': coco_eval.stats[6],    # AR given 1 detection per image
                'AR_10': coco_eval.stats[7],   # AR given 10 detections per image
                'AR_100': coco_eval.stats[8],  # AR given 100 detections per image
                'AR_small': coco_eval.stats[9],    # AR for small objects
                'AR_medium': coco_eval.stats[10],  # AR for medium objects
                'AR_large': coco_eval.stats[11],   # AR for large objects
            }
            
            # Save detailed results
            results_path = os.path.join(self.output_dir, 'evaluation', f'coco_results{nms_suffix}.json')
            with open(results_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"COCO evaluation results {'with' if self.enable_nms else 'without'} NMS:")
            print(f"  AP: {metrics['AP']:.4f}")
            print(f"  AP@0.5: {metrics['AP_50']:.4f}")
            print(f"  AP@0.75: {metrics['AP_75']:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error during COCO evaluation: {e}")
            return {}


def collate_fn(batch):
    """Custom collate function for batch processing"""
    patches = torch.stack([item['patches'] for item in batch])
    targets = [item['targets'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    modality = batch[0]['modality']  # All items in batch should have same modality
    
    return {
        'patches': patches,
        'targets': targets,
        'image_ids': image_ids,
        'modality': modality
    }


def train_epoch(model, dataloader, criterion, optimizer, device, args, logger):
    """Train for one epoch with optional NMS-based loss"""
    model.train()
    total_loss = 0
    total_batches = 0
    total_bbox_loss = 0
    total_giou_loss = 0
    total_ce_loss = 0
    total_original_preds = 0
    total_filtered_preds = 0
    
    mode_desc = f"Training ({args.modality})" + (" + NMS" if args.train_with_nms else "")
    pbar = tqdm(dataloader, desc=mode_desc)
    
    for batch_idx, batch in enumerate(pbar):
        patches = batch['patches'].to(device)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v 
                   for k, v in t.items()} for t in batch['targets']]
        
        # Debug: Print target information for first batch
        if batch_idx == 0:
            total_targets = sum(len(t['labels']) for t in targets)
            print(f"First batch - Total targets: {total_targets}")
            if total_targets > 0:
                print(f"Sample target boxes: {targets[0]['boxes'][:3] if len(targets[0]['boxes']) > 0 else 'No boxes'}")
                print(f"Sample target labels: {targets[0]['labels'][:3] if len(targets[0]['labels']) > 0 else 'No labels'}")
            if args.train_with_nms:
                print(f"Training with NMS - threshold: {args.train_nms_threshold}, max_det: {args.train_max_detections}")
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(patches)
        
        # Debug: Print output shapes for first batch
        if batch_idx == 0:
            print(f"Output shapes - pred_boxes: {outputs['pred_boxes'].shape}, pred_logits: {outputs['pred_logits'].shape}")
            print(f"Pred boxes range: [{outputs['pred_boxes'].min():.3f}, {outputs['pred_boxes'].max():.3f}]")
            print(f"Pred logits range: [{outputs['pred_logits'].min():.3f}, {outputs['pred_logits'].max():.3f}]")
        
        # Compute loss - with or without NMS filtering
        if args.train_with_nms:
            # Apply NMS filtering to predictions before loss computation
            filtered_info = filter_predictions_for_training(
                outputs,
                confidence_threshold=args.train_confidence_threshold,
                nms_threshold=args.train_nms_threshold,
                max_detections=args.train_max_detections,
                num_classes=args.num_classes
            )
            
            # Track filtering statistics
            stats = filtered_info['statistics']
            total_original_preds += stats['total_original']
            total_filtered_preds += stats['total_filtered']
            
            if batch_idx == 0:
                print(f"NMS filtering: {stats['total_original']} -> {stats['total_filtered']} predictions "
                      f"(ratio: {stats['overall_filter_ratio']:.3f})")
            
            # Compute loss on filtered predictions
            filtered_outputs = {
                'pred_boxes': filtered_info['pred_boxes'],
                'pred_logits': filtered_info['pred_logits']
            }
            
            loss_dict = criterion(filtered_outputs, targets, filtered_info, outputs)
        else:
            # Standard loss computation without NMS
            loss_dict = criterion(outputs, targets)
        
        loss = loss_dict['total_loss']
        
        # Check for invalid loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss detected at batch {batch_idx}: {loss}")
            continue
            
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_bbox_loss += loss_dict['loss_bbox'].item()
        total_giou_loss += loss_dict['loss_giou'].item()
        total_ce_loss += loss_dict['loss_ce'].item()
        total_batches += 1
        
        # Update progress bar
        postfix = {
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss/total_batches:.4f}",
            'bbox': f"{loss_dict['loss_bbox']:.4f}",
            'giou': f"{loss_dict['loss_giou']:.4f}",
            'ce': f"{loss_dict['loss_ce']:.4f}"
        }
        
        if args.train_with_nms and total_original_preds > 0:
            filter_ratio = total_filtered_preds / total_original_preds
            postfix['filter'] = f"{filter_ratio:.3f}"
        
        pbar.set_postfix(postfix)
        
        # Log batch-level metrics to file
        if batch_idx % args.log_interval == 0:
            batch_log = {
                'batch': batch_idx,
                'total_loss': loss.item(),
                'bbox_loss': loss_dict['loss_bbox'].item(),
                'giou_loss': loss_dict['loss_giou'].item(),
                'ce_loss': loss_dict['loss_ce'].item(),
                'train_with_nms': args.train_with_nms
            }
            
            if args.train_with_nms:
                batch_log.update({
                    'nms_threshold': args.train_nms_threshold,
                    'confidence_threshold': args.train_confidence_threshold,
                    'filter_ratio': total_filtered_preds / total_original_preds if total_original_preds > 0 else 0
                })
            
            batch_log_path = os.path.join(logger.output_dir, 'logs', 'batch_logs.jsonl')
            with open(batch_log_path, 'a') as f:
                f.write(json.dumps(batch_log) + '\n')
    
    avg_total_loss = total_loss / total_batches if total_batches > 0 else 0
    avg_bbox_loss = total_bbox_loss / total_batches if total_batches > 0 else 0
    avg_giou_loss = total_giou_loss / total_batches if total_batches > 0 else 0
    avg_ce_loss = total_ce_loss / total_batches if total_batches > 0 else 0
    
    # Print epoch summary
    if args.train_with_nms:
        filter_ratio = total_filtered_preds / total_original_preds if total_original_preds > 0 else 0
        print(f"Epoch averages - Total: {avg_total_loss:.4f}, BBox: {avg_bbox_loss:.4f}, "
              f"GIoU: {avg_giou_loss:.4f}, CE: {avg_ce_loss:.4f}, Filter Ratio: {filter_ratio:.3f}")
    else:
        print(f"Epoch averages - Total: {avg_total_loss:.4f}, BBox: {avg_bbox_loss:.4f}, "
              f"GIoU: {avg_giou_loss:.4f}, CE: {avg_ce_loss:.4f}")
    
    metrics = {
        'total_loss': avg_total_loss,
        'bbox_loss': avg_bbox_loss,
        'giou_loss': avg_giou_loss,
        'ce_loss': avg_ce_loss
    }
    
    if args.train_with_nms:
        metrics['filter_ratio'] = filter_ratio
        metrics['original_predictions'] = total_original_preds
        metrics['filtered_predictions'] = total_filtered_preds
    
    return metrics


def validate_epoch(model, dataloader, criterion, device, modality):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_batches = 0
    total_bbox_loss = 0
    total_giou_loss = 0
    total_ce_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Validation ({modality})")
        for batch_idx, batch in enumerate(pbar):
            patches = batch['patches'].to(device)
            targets = [{k: v.to(device) if torch.is_tensor(v) else v 
                       for k, v in t.items()} for t in batch['targets']]
            
            # Forward pass
            outputs = model(patches)
            
            # Compute loss
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total_loss']
            
            total_loss += loss.item()
            total_bbox_loss += loss_dict['loss_bbox'].item()
            total_giou_loss += loss_dict['loss_giou'].item()
            total_ce_loss += loss_dict['loss_ce'].item()
            total_batches += 1
            
            pbar.set_postfix({'val_loss': f"{total_loss/total_batches:.4f}"})
    
    avg_total_loss = total_loss / total_batches if total_batches > 0 else 0
    avg_bbox_loss = total_bbox_loss / total_batches if total_batches > 0 else 0
    avg_giou_loss = total_giou_loss / total_batches if total_batches > 0 else 0
    avg_ce_loss = total_ce_loss / total_batches if total_batches > 0 else 0
    
    print(f"Val averages - Total: {avg_total_loss:.4f}, BBox: {avg_bbox_loss:.4f}, GIoU: {avg_giou_loss:.4f}, CE: {avg_ce_loss:.4f}")
    
    return {
        'total_loss': avg_total_loss,
        'bbox_loss': avg_bbox_loss,
        'giou_loss': avg_giou_loss,
        'ce_loss': avg_ce_loss
    }


def main():
    args = parse_args()
    
    # Create output directory with detailed parameters
    output_dir = create_output_directory(args)
    
    # Compute actual hidden size based on modality
    if args.modality == 'concat':
        hidden_size = args.base_hidden_size * 2
        print(f"Using concatenated modality: RGB + IR, hidden_size = {hidden_size}")
    else:
        hidden_size = args.base_hidden_size
        print(f"Using {args.modality} modality, hidden_size = {hidden_size}")
    
    # Print NMS configuration
    print(f"\n=== NMS Configuration ===")
    if args.train_with_nms:
        print(f"Training NMS: ENABLED")
        print(f"  - NMS threshold: {args.train_nms_threshold}")
        print(f"  - Confidence threshold: {args.train_confidence_threshold}")
        print(f"  - Max detections: {args.train_max_detections}")
    else:
        print(f"Training NMS: DISABLED (standard training)")
    
    if args.enable_nms:
        print(f"Evaluation NMS: ENABLED")
        print(f"  - NMS threshold: {args.nms_threshold}")
        print(f"  - Confidence threshold: {args.confidence_threshold}")
        print(f"  - Max detections: {args.max_detections}")
    else:
        print(f"Evaluation NMS: DISABLED")
    print("="*30)
    
    # Initialize metrics logger
    logger = MetricsLogger(output_dir)
    
    # Load features
    print(f"Loading features from {args.features_path}")
    with open(args.features_path, 'rb') as f:
        features_data = pickle.load(f)
    
    # Check if we have the new combined train/val structure
    if 'train' in features_data and 'val' in features_data:
        print("Found combined train/val features structure")
        train_features = features_data['train']
        val_features = features_data['val']
    else:
        # Try to load separate files or use legacy structure
        print("Legacy features structure detected. Looking for separate train/val files...")
        train_path = args.features_path.replace('.pkl', '_train.pkl')
        val_path = args.features_path.replace('.pkl', '_val.pkl')
        
        if os.path.exists(train_path) and os.path.exists(val_path):
            print(f"Loading separate train file: {train_path}")
            with open(train_path, 'rb') as f:
                train_features = pickle.load(f)
            print(f"Loading separate val file: {val_path}")
            with open(val_path, 'rb') as f:
                val_features = pickle.load(f)
        else:
            print("ERROR: Could not find proper train/val split structure!")
            print("Available keys in features file:", list(features_data.keys()))
            print("\nPlease extract features with both train and val splits:")
            print("python main.py --mode extract_features --extract_patch_embeddings")
            raise ValueError("Train/val splits not found. Please re-extract features with both splits.")
    
    # Verify that we have patch embeddings
    for split_name, split_data in [('train', train_features), ('val', val_features)]:
        if 'rgb_patch_embeddings' not in split_data or 'ir_patch_embeddings' not in split_data:
            print(f"\nERROR: Patch embeddings not found in {split_name} data!")
            print(f"Available keys in {split_name}:", list(split_data.keys()))
            print("\nPlease extract features with patch embeddings first:")
            print("python main.py --mode extract_features --extract_patch_embeddings")
            raise ValueError(f"Patch embeddings not found in {split_name} data.")
    
    print(f"Loaded train features for {len(train_features['metadata'])} images")
    print(f"Loaded val features for {len(val_features['metadata'])} images")
    print(f"RGB patch embeddings shape: {train_features['rgb_patch_embeddings'].shape}")
    print(f"IR patch embeddings shape: {train_features['ir_patch_embeddings'].shape}")
    
    # Validate shapes
    train_rgb_shape = train_features['rgb_patch_embeddings'].shape
    train_ir_shape = train_features['ir_patch_embeddings'].shape
    val_rgb_shape = val_features['rgb_patch_embeddings'].shape
    val_ir_shape = val_features['ir_patch_embeddings'].shape
    
    if train_rgb_shape[1:] != val_rgb_shape[1:] or train_ir_shape[1:] != val_ir_shape[1:]:
        print(f"WARNING: Train and val embeddings have different shapes!")
        print(f"Train RGB: {train_rgb_shape}, Val RGB: {val_rgb_shape}")
        print(f"Train IR: {train_ir_shape}, Val IR: {val_ir_shape}")
    
    if train_rgb_shape[-1] != args.base_hidden_size:
        print(f"WARNING: Expected hidden size {args.base_hidden_size}, but got {train_rgb_shape[-1]}")
        print("Updating base_hidden_size to match the data...")
        args.base_hidden_size = train_rgb_shape[-1]
        
        # Recompute hidden size
        if args.modality == 'concat':
            hidden_size = args.base_hidden_size * 2
        else:
            hidden_size = args.base_hidden_size
    
    # Create datasets using proper train/val splits
    print(f"Creating datasets for {args.modality} modality...")
    train_dataset = PatchEmbeddingDataset(train_features, modality=args.modality)
    val_dataset = PatchEmbeddingDataset(val_features, modality=args.modality)
    
    # Analyze dataset statistics
    print(f"\nDataset Statistics:")
    total_train_objects = 0
    total_val_objects = 0
    train_class_counts = {}
    val_class_counts = {}
    
    for i in range(min(100, len(train_dataset))):  # Sample first 100 for stats
        sample = train_dataset[i]
        labels = sample['targets']['labels']
        total_train_objects += len(labels)
        for label in labels:
            label_name = list(train_dataset.category_mapping.keys())[label.item()]
            train_class_counts[label_name] = train_class_counts.get(label_name, 0) + 1
    
    for i in range(min(100, len(val_dataset))):  # Sample first 100 for stats
        sample = val_dataset[i]
        labels = sample['targets']['labels']
        total_val_objects += len(labels)
        for label in labels:
            label_name = list(val_dataset.category_mapping.keys())[label.item()]
            val_class_counts[label_name] = val_class_counts.get(label_name, 0) + 1
    
    print(f"Sample train objects (first 100 images): {total_train_objects}")
    print(f"Sample val objects (first 100 images): {total_val_objects}")
    print(f"Train class distribution: {train_class_counts}")
    print(f"Val class distribution: {val_class_counts}")
    
    # Save dataset statistics
    dataset_stats = {
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'train_objects_sample': total_train_objects,
        'val_objects_sample': total_val_objects,
        'train_class_distribution': train_class_counts,
        'val_class_distribution': val_class_counts,
        'category_mapping': train_dataset.category_mapping,
        'training_nms_config': {
            'train_with_nms': args.train_with_nms,
            'train_nms_threshold': args.train_nms_threshold,
            'train_confidence_threshold': args.train_confidence_threshold,
            'train_max_detections': args.train_max_detections
        },
        'evaluation_nms_config': {
            'enable_nms': args.enable_nms,
            'nms_threshold': args.nms_threshold,
            'confidence_threshold': args.confidence_threshold,
            'max_detections': args.max_detections
        }
    }
    
    with open(os.path.join(output_dir, 'dataset_statistics.json'), 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    print(f"Train samples: {len(train_dataset)}, Train batches: {len(train_dataloader)}")
    print(f"Val samples: {len(val_dataset)}, Val batches: {len(val_dataloader)}")
    
    # Create model with correct hidden size
    model = SigLIPDetectionModel(
        hidden_size=hidden_size,
        num_classes=args.num_classes
    ).to(args.device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Input hidden size: {hidden_size}")
    
    # Create loss function
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    criterion = DetectionLoss(
        num_classes=args.num_classes,
        matcher=matcher,
        bbox_loss_coef=args.bbox_loss_coef,
        giou_loss_coef=args.giou_loss_coef,
        class_loss_coef=args.class_loss_coef
    )
    
    # Create COCO evaluator with NMS support
    evaluator = COCOEvaluator(
        output_dir, 
        train_dataset.category_mapping,
        enable_nms=args.enable_nms,
        nms_threshold=args.nms_threshold,
        max_detections=args.max_detections
    )
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    # Create warmup + cosine annealing scheduler
    warmup_epochs = args.warmup_epochs
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            # Cosine annealing after warmup
            return 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup_epochs) / (args.num_epochs - warmup_epochs)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"Starting training for {args.modality} modality...")
    print(f"Warmup epochs: {args.warmup_epochs}, Total epochs: {args.num_epochs}")
    print(f"Number of classes: {args.num_classes} + 1 (background) = {args.num_classes + 1}")
    print(f"Class mapping: {train_dataset.category_mapping}")
    print(f"Output directory: {output_dir}")
    
    best_val_loss = float('inf')
    best_ap = 0.0
    
    # Debug: Check first few samples
    sample_batch = next(iter(train_dataloader))
    print(f"Sample batch info:")
    print(f"  Patches shape: {sample_batch['patches'].shape}")
    print(f"  Number of targets: {len(sample_batch['targets'])}")
    total_objects = sum(len(t['labels']) for t in sample_batch['targets'])
    print(f"  Total objects in sample batch: {total_objects}")
    if total_objects > 0:
        sample_labels = [t['labels'].tolist() for t in sample_batch['targets'] if len(t['labels']) > 0][:3]
        print(f"  Sample labels: {sample_labels}")
    
    for epoch in range(args.num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{args.num_epochs} - {args.modality.upper()} Modality")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Train
        train_metrics = train_epoch(model, train_dataloader, criterion, optimizer, args.device, args, logger)
        
        # Validate
        val_metrics = validate_epoch(model, val_dataloader, criterion, args.device, args.modality)
        
        # Step scheduler
        scheduler.step()
        
        # Run COCO evaluation
        ap_metrics = {}
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.num_epochs - 1:
            print(f"Running COCO evaluation...")
            ap_metrics = evaluator.evaluate(model, val_dataloader, args.device, args.confidence_threshold)
            if ap_metrics:
                print(f"AP: {ap_metrics['AP']:.4f}, AP@0.5: {ap_metrics['AP_50']:.4f}, AP@0.75: {ap_metrics['AP_75']:.4f}")
                print(f"AR@100: {ap_metrics['AR_100']:.4f}")
        
        # Log metrics
        logger.log_epoch(epoch, train_metrics, val_metrics, current_lr, ap_metrics)
        
        print(f"Train Loss: {train_metrics['total_loss']:.4f}, Val Loss: {val_metrics['total_loss']:.4f}")
        if ap_metrics:
            print(f"AP: {ap_metrics.get('AP', 0):.4f}")
        
        # Update best metrics
        is_best_loss = val_metrics['total_loss'] < best_val_loss
        is_best_ap = ap_metrics.get('AP', 0) > best_ap
        
        if is_best_loss:
            best_val_loss = val_metrics['total_loss']
        if ap_metrics.get('AP', 0) > 0:
            best_ap = max(best_ap, ap_metrics['AP'])
        
        # Save best models
        if is_best_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['total_loss'],
                'train_loss': train_metrics['total_loss'],
                'ap_metrics': ap_metrics,
                'modality': args.modality,
                'hidden_size': hidden_size,
                'training_nms_config': {
                    'train_with_nms': args.train_with_nms,
                    'train_nms_threshold': args.train_nms_threshold,
                    'train_confidence_threshold': args.train_confidence_threshold,
                    'train_max_detections': args.train_max_detections
                },
                'evaluation_nms_config': {
                    'enable_nms': args.enable_nms,
                    'nms_threshold': args.nms_threshold,
                    'confidence_threshold': args.confidence_threshold,
                    'max_detections': args.max_detections
                },
                'args': vars(args)
            }, os.path.join(output_dir, 'models', 'best_model_loss.pth'))
            print(f"Saved new best model (loss) with val_loss: {val_metrics['total_loss']:.4f}")
        
        if is_best_ap and ap_metrics:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['total_loss'],
                'train_loss': train_metrics['total_loss'],
                'ap_metrics': ap_metrics,
                'modality': args.modality,
                'hidden_size': hidden_size,
                'training_nms_config': {
                    'train_with_nms': args.train_with_nms,
                    'train_nms_threshold': args.train_nms_threshold,
                    'train_confidence_threshold': args.train_confidence_threshold,
                    'train_max_detections': args.train_max_detections
                },
                'evaluation_nms_config': {
                    'enable_nms': args.enable_nms,
                    'nms_threshold': args.nms_threshold,
                    'confidence_threshold': args.confidence_threshold,
                    'max_detections': args.max_detections
                },
                'args': vars(args)
            }, os.path.join(output_dir, 'models', 'best_model_ap.pth'))
            print(f"Saved new best model (AP) with AP: {ap_metrics['AP']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['total_loss'],
                'train_loss': train_metrics['total_loss'],
                'ap_metrics': ap_metrics,
                'modality': args.modality,
                'hidden_size': hidden_size,
                'training_nms_config': {
                    'train_with_nms': args.train_with_nms,
                    'train_nms_threshold': args.train_nms_threshold,
                    'train_confidence_threshold': args.train_confidence_threshold,
                    'train_max_detections': args.train_max_detections
                },
                'evaluation_nms_config': {
                    'enable_nms': args.enable_nms,
                    'nms_threshold': args.nms_threshold,
                    'confidence_threshold': args.confidence_threshold,
                    'max_detections': args.max_detections
                },
                'args': vars(args)
            }, os.path.join(output_dir, 'models', f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Save and plot metrics periodically
        if (epoch + 1) % 5 == 0:
            logger.save_metrics()
            logger.plot_metrics()
    
    # Final save and plot
    logger.save_metrics()
    logger.plot_metrics()
    
    print(f"Training completed for {args.modality} modality!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best AP: {best_ap:.4f}")
    print(f"Results saved in: {output_dir}")
    
    # Print final configuration summary
    print(f"\n=== Final Configuration Summary ===")
    print(f"Training method: {'NMS-filtered loss' if args.train_with_nms else 'Standard loss'}")
    if args.train_with_nms:
        print(f"  Training NMS threshold: {args.train_nms_threshold}")
        print(f"  Training confidence threshold: {args.train_confidence_threshold}")
    
    evaluation_method = "with NMS" if args.enable_nms else "without NMS"
    print(f"Evaluation method: {evaluation_method}")
    if args.enable_nms:
        print(f"  Evaluation NMS threshold: {args.nms_threshold}")
        print(f"  Evaluation confidence threshold: {args.confidence_threshold}")
    print("="*40)


if __name__ == "__main__":
    main()