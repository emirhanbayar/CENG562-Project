import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import os

# === Configuration ===
# Update these paths accordingly
ground_truth_path = '/home/emirhan/datasets/object_detection/coco-mini-class-agnostic/annotations/instances_val2017.json'  # Path to COCO ground truth annotations
# =====================

def main():
    # Initialize COCO ground truth
    print("Loading ground truth annotations...")
    with open(ground_truth_path, 'r') as f:
        coco_gt = json.load(f)

    # print(coco_gt['annotations'][0])
    for ann in coco_gt['annotations']:
        ann['category_id'] = 1

    # Save predictions with updated category_id
    ground_truth_output_path = "/home/emirhan/datasets/object_detection/coco-mini-class-agnostic/annotations/instances_val2017.json"

    with open(ground_truth_output_path, 'w') as f:
        json.dump(coco_gt, f)

    print(f"Saved updated ground truth annotations to {ground_truth_output_path}")

if __name__ == '__main__':
    main()