import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
from PIL import Image
import os
import random

class DetectionVisualizer:
    def __init__(self, ann_path, pred_path, img_dir):
        """
        Initialize the detection visualizer.
        
        Args:
            ann_path (str): Path to COCO annotation file
            pred_path (str): Path to prediction JSON file
            img_dir (str): Directory containing the images
        """
        self.coco = COCO(ann_path)
        with open(pred_path, 'r') as f:
            self.predictions = json.load(f).get('predictions', [])
        self.img_dir = img_dir
        
        # Create color map for categories
        categories = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
        self.colors = plt.cm.rainbow(np.linspace(0, 1, len(categories)))
        self.cat_id_to_color = {cat['id']: self.colors[i] for i, cat in enumerate(categories)}
        
        # Create output directories
        self.output_dirs = {
            'all': 'all_detections',
            'tp': 'true_positives',
            'fp': 'false_positives',
            'fn': 'false_negatives'
        }
        for dir_name in self.output_dirs.values():
            os.makedirs(dir_name, exist_ok=True)

    def draw_bbox(self, ax, bbox, color, is_prediction=False, is_matched=None):
        """Draw a bounding box on the axes with different styles based on match status."""
        x, y, w, h = bbox
        linestyle = '--' if is_prediction else '-'
        if is_matched is not None:
            linewidth = 3 if is_matched else 1
        else:
            linewidth = 2
            
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=linewidth,
            edgecolor=color,
            facecolor='none',
            linestyle=linestyle
        )
        ax.add_patch(rect)

    def compute_iou(self, bbox1, bbox2):
        """Compute IoU between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to xmin, ymin, xmax, ymax
        xmin1, ymin1, xmax1, ymax1 = x1, y1, x1 + w1, y1 + h1
        xmin2, ymin2, xmax2, ymax2 = x2, y2, x2 + w2, y2 + h2
        
        # Compute intersection
        xmin_inter = max(xmin1, xmin2)
        ymin_inter = max(ymin1, ymin2)
        xmax_inter = min(xmax1, xmax2)
        ymax_inter = min(ymax1, ymax2)
        
        if xmax_inter <= xmin_inter or ymax_inter <= ymin_inter:
            return 0.0
        
        inter_area = (xmax_inter - xmin_inter) * (ymax_inter - ymin_inter)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area

    def match_detections(self, gt_anns, pred_anns, iou_threshold=0.5):
        """Match predictions to ground truth annotations following COCO matching strategy."""
        matches = []  # [(pred_idx, gt_idx, iou)]
        unmatched_preds = list(range(len(pred_anns)))
        unmatched_gts = list(range(len(gt_anns)))
        
        if len(gt_anns) == 0 or len(pred_anns) == 0:
            return matches, unmatched_preds, unmatched_gts

        # Calculate IoU matrix
        ious = np.zeros((len(pred_anns), len(gt_anns)))
        for i, pred in enumerate(pred_anns):
            for j, gt in enumerate(gt_anns):
                if gt['category_id'] == pred['category_id']:
                    ious[i, j] = self.compute_iou(pred['bbox'], gt['bbox'])
                else:
                    ious[i, j] = 0.0

        # Sort predictions by score
        pred_scores = np.array([-p['score'] for p in pred_anns])
        pred_sorted_idx = np.argsort(pred_scores, kind='mergesort')
        
        # Initialize arrays for matched gt and predictions
        matched_gt = np.zeros(len(gt_anns), dtype=bool)
        matched_pred = np.zeros(len(pred_anns), dtype=bool)

        # For each prediction in descending score order
        for pred_idx in pred_sorted_idx:
            # Find the best matching ground truth for this prediction
            iou = ious[pred_idx]
            # Only consider unmatched ground truths
            iou[matched_gt] = 0
            best_match_idx = np.argmax(iou)
            best_match_iou = iou[best_match_idx]

            if best_match_iou >= iou_threshold:
                matches.append((pred_idx, best_match_idx, best_match_iou))
                matched_gt[best_match_idx] = True
                matched_pred[pred_idx] = True

        unmatched_preds = np.where(~matched_pred)[0].tolist()
        unmatched_gts = np.where(~matched_gt)[0].tolist()

        return matches, unmatched_preds, unmatched_gts

    def visualize_errors(self, img_id, conf_threshold=0.5):
        """Visualize only false positives and false negatives for error analysis."""
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path)
        
        # Get ground truth annotations
        gt_anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        pred_anns = [p for p in self.predictions if p['image_id'] == img_id and p['score'] > conf_threshold]
        
        # Match detections
        matches, unmatched_preds, unmatched_gts = self.match_detections(gt_anns, pred_anns)
        
        # Only create visualization if there are errors
        if unmatched_preds or unmatched_gts:
            fig, ax = plt.subplots(1, 1, figsize=(15, 15))
            ax.imshow(img)
            
            # Initialize label positions tracker
            label_positions = []  # List of (x, y, width, height) of label bboxes
            
            def find_non_overlapping_position(x, y, label_width, label_height):
                """Find a position for the label that doesn't overlap with existing labels."""
                original_y = y
                y_offset = 5  # Start with small offset
                max_attempts = 50  # Prevent infinite loop
                
                for _ in range(max_attempts):
                    # Try position above the box
                    test_y = y - y_offset
                    
                    # Check if this position overlaps with any existing label
                    overlap = False
                    for lx, ly, lw, lh in label_positions:
                        if (x < lx + lw and x + label_width > lx and
                            test_y < ly + lh and test_y + label_height > ly):
                            overlap = True
                            break
                    
                    if not overlap:
                        return test_y
                    
                    # Try position below the box
                    test_y = original_y + y_offset
                    
                    # Check if this position overlaps
                    overlap = False
                    for lx, ly, lw, lh in label_positions:
                        if (x < lx + lw and x + label_width > lx and
                            test_y < ly + lh and test_y + label_height > ly):
                            overlap = True
                            break
                    
                    if not overlap:
                        return test_y
                    
                    y_offset += 20  # Increment offset and try again
                
                return y - 5  # Fallback to default position if no solution found

            # Draw unmatched predictions (false positives)
            for pred_idx in unmatched_preds:
                pred = pred_anns[pred_idx]
                color = self.cat_id_to_color[pred['category_id']]
                self.draw_bbox(ax, pred['bbox'], color, is_prediction=True, is_matched=False)
                
                # Calculate label position
                x, y = pred['bbox'][0], pred['bbox'][1]
                label = f"{self.cat_id_to_name[pred['category_id']]} ({pred['score']:.2f}) [FP]"
                
                # Estimate text size (approximate)
                label_width = len(label) * 7  # Approximate width based on character count
                label_height = 20  # Approximate height
                
                # Find non-overlapping position
                y = find_non_overlapping_position(x, y, label_width, label_height)
                
                # Add text and update label positions
                ax.text(x, y, label, color=color, fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.7))
                label_positions.append((x, y, label_width, label_height))

            # Draw unmatched ground truths (false negatives)
            for gt_idx in unmatched_gts:
                gt = gt_anns[gt_idx]
                color = self.cat_id_to_color[gt['category_id']]
                self.draw_bbox(ax, gt['bbox'], color, is_prediction=False, is_matched=False)
                
                # Calculate label position
                x, y = gt['bbox'][0], gt['bbox'][1]
                label = f"{self.cat_id_to_name[gt['category_id']]} [FN]"
                
                # Estimate text size
                label_width = len(label) * 7
                label_height = 20
                
                # Find non-overlapping position
                y = find_non_overlapping_position(x, y, label_width, label_height)
                
                # Add text and update label positions
                ax.text(x, y, label, color=color, fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.7))
                label_positions.append((x, y, label_width, label_height))

            ax.axis('off')
            plt.title("Classification Errors (FP: False Positives, FN: False Negatives)", pad=20)
            
            # Save to errors directory
            os.makedirs('classification_errors', exist_ok=True)
            plt.savefig(os.path.join('classification_errors', f'errors_{img_id}.png'),
                       bbox_inches='tight', dpi=300)
            plt.close()

    def visualize_image(self, img_id, conf_threshold=0.5):
        """Visualize and save detections for a single image."""
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path)
        
        # Get ground truth annotations
        gt_anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        pred_anns = [p for p in self.predictions if p['image_id'] == img_id and p['score'] > conf_threshold]
        
        # Match detections
        matches, unmatched_preds, unmatched_gts = self.match_detections(gt_anns, pred_anns)

        # Create separate visualizations for GT and predictions in 'all' directory
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
        
        # Ground Truth visualization
        ax1.imshow(img)
        ax1.set_title('Ground Truth', fontsize=16)
        for gt_idx, gt in enumerate(gt_anns):
            color = self.cat_id_to_color[gt['category_id']]
            self.draw_bbox(ax1, gt['bbox'], color, is_prediction=False)
            x, y = gt['bbox'][0], gt['bbox'][1]
            label = f"{self.cat_id_to_name[gt['category_id']]}"
            ax1.text(x, y-5, label, color=color, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7))
        ax1.axis('off')

        # Predictions visualization
        ax2.imshow(img)
        ax2.set_title('Predictions', fontsize=16)
        for pred_idx, pred in enumerate(pred_anns):
            color = self.cat_id_to_color[pred['category_id']]
            is_matched = any(m[0] == pred_idx for m in matches)
            self.draw_bbox(ax2, pred['bbox'], color, is_prediction=True)
            x, y = pred['bbox'][0], pred['bbox'][1]
            label = f"{self.cat_id_to_name[pred['category_id']]} ({pred['score']:.2f})"
            ax2.text(x, y-5, label, color=color, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7))
        ax2.axis('off')

        # Save to 'all' directory
        plt.savefig(os.path.join(self.output_dirs['all'], f'detection_{img_id}.png'),
                   bbox_inches='tight', dpi=300)
        plt.close()

        # Create single visualization for other directories
        if matches or unmatched_preds or unmatched_gts:
            fig, ax = plt.subplots(1, 1, figsize=(15, 15))
            ax.imshow(img)

            # Draw matched pairs
            for pred_idx, gt_idx, iou in matches:
                pred = pred_anns[pred_idx]
                gt = gt_anns[gt_idx]
                color = self.cat_id_to_color[pred['category_id']]
                
                self.draw_bbox(ax, gt['bbox'], color, is_prediction=False, is_matched=True)
                self.draw_bbox(ax, pred['bbox'], color, is_prediction=True, is_matched=True)
                
                x, y = pred['bbox'][0], pred['bbox'][1]
                label = f"{self.cat_id_to_name[pred['category_id']]} ({pred['score']:.2f})"
                ax.text(x, y-5, label, color=color, fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7))

            # Draw unmatched predictions (false positives)
            for pred_idx in unmatched_preds:
                pred = pred_anns[pred_idx]
                color = self.cat_id_to_color[pred['category_id']]
                self.draw_bbox(ax, pred['bbox'], color, is_prediction=True, is_matched=False)
                x, y = pred['bbox'][0], pred['bbox'][1]
                label = f"{self.cat_id_to_name[pred['category_id']]} ({pred['score']:.2f}) [FP]"
                ax.text(x, y-5, label, color=color, fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7))

            # Draw unmatched ground truths (false negatives)
            for gt_idx in unmatched_gts:
                gt = gt_anns[gt_idx]
                color = self.cat_id_to_color[gt['category_id']]
                self.draw_bbox(ax, gt['bbox'], color, is_prediction=False, is_matched=False)
                x, y = gt['bbox'][0], gt['bbox'][1]
                label = f"{self.cat_id_to_name[gt['category_id']]} [FN]"
                ax.text(x, y-5, label, color=color, fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7))

            ax.axis('off')

            # Save to specific directories based on content
            base_filename = f'detection_{img_id}.png'
            if matches:  # Has true positives
                plt.savefig(os.path.join(self.output_dirs['tp'], base_filename),
                           bbox_inches='tight', dpi=300)
            if unmatched_preds:  # Has false positives
                plt.savefig(os.path.join(self.output_dirs['fp'], base_filename),
                           bbox_inches='tight', dpi=300)
            if unmatched_gts:  # Has false negatives
                plt.savefig(os.path.join(self.output_dirs['fn'], base_filename),
                           bbox_inches='tight', dpi=300)

            plt.close()

    def visualize_dataset(self, num_samples=None, conf_threshold=0.5):
        """
        Visualize multiple images from the dataset.
        
        Args:
            num_samples (int): Number of random samples to visualize. If None, visualize all.
            conf_threshold (float): Confidence threshold for predictions
        """
        img_ids = self.coco.getImgIds()
        if num_samples is not None:
            img_ids = random.sample(img_ids, min(num_samples, len(img_ids)))
        
        for i, img_id in enumerate(img_ids):
            print(f"Processing image {i+1}/{len(img_ids)} (ID: {img_id})")
            self.visualize_image(img_id, conf_threshold)
            self.visualize_errors(img_id, conf_threshold)
        """
        Visualize multiple images from the dataset.
        
        Args:
            num_samples (int): Number of random samples to visualize. If None, visualize all.
            conf_threshold (float): Confidence threshold for predictions
        """
        img_ids = self.coco.getImgIds()
        if num_samples is not None:
            img_ids = random.sample(img_ids, min(num_samples, len(img_ids)))
        
        for i, img_id in enumerate(img_ids):
            print(f"Processing image {i+1}/{len(img_ids)} (ID: {img_id})")
            self.visualize_image(img_id, conf_threshold)

def main():
    # Configuration
    ann_path = '/home/emirhan/datasets/object_detection/coco/annotations/instances_val2017.json'
    pred_path = '/home/emirhan/deteval/predictions_AR_maximize_experiment.json'
    img_dir = '/home/emirhan/datasets/object_detection/coco/val2017'
    
    # Initialize visualizer
    visualizer = DetectionVisualizer(ann_path, pred_path, img_dir)
    
    # Visualize dataset
    visualizer.visualize_dataset(num_samples=100, conf_threshold=0.0)
    
    print("Visualization complete. Results saved in the following directories:")
    for dir_name in visualizer.output_dirs.values():
        print(f"- {dir_name}")

if __name__ == "__main__":
    main()