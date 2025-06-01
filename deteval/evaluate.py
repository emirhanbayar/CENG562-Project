import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import os

# === Configuration ===
# Update these paths accordingly
# ground_truth_path = '/home/emirhan/datasets/object_detection/coco/annotations/instances_val2017.json'  # Path to COCO ground truth annotations
# predictions_path = '/home/emirhan/DINO-GT-Cheat/logs/DINO/R50-MS4-%j/predictions_original.json'         # Path to your predictions.json file

# ground_truth_path = '/home/emirhan/datasets/object_detection/coco/annotations/instances_val2017.json'
# predictions_path = '/home/emirhan/Deformable-DETR/logs_original/predictions.json'

# img_dir = '/home/emirhan/datasets/object_detection/coco/val2017'

ground_truth_path = '/home/emirhan/datasets/object_detection/FLIR_ADAS_IR/annotations/instances_val2017_remapped.json'
predictions_path = '/home/emirhan/DINO_RGB_IR/logs/DINO/R50-MS4-RGB/predictions.json'
img_dir = '/home/emirhan/datasets/object_detection/FLIR_ADAS_RGB/val2017/'

# =====================
# =====================

def main():
    # Initialize COCO ground truth
    print("Loading ground truth annotations...")
    coco_gt = COCO(ground_truth_path)

    # Load predictions
    print("Loading predictions...")
    with open(predictions_path, 'r') as f:
        predictions_data = json.load(f)
        predictions = predictions_data.get('predictions', [])

    if not predictions:
        print("No predictions found in the predictions.json file.")
        return

    # Convert predictions to COCO format
    print("Converting predictions to COCO format...")
    coco_dt = coco_gt.loadRes(predictions)

    # Initialize COCOeval for bbox
    print("Initializing COCO evaluation for bounding boxes...")
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox', img_dir=img_dir)

    # Run evaluation
    print("Running evaluation...")
    coco_eval.params.useCats = 1  # Evaluate all categories
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
if __name__ == '__main__':
    main()