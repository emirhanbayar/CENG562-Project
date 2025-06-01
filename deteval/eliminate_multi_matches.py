import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict
import copy

def get_multi_match_image_ids(eval_imgs):
    """Extract image IDs that have multi-match detections"""
    multi_match_img_ids = set()
    
    for eval_img in eval_imgs:
        if eval_img is None:
            continue
            
        img_id = eval_img['image_id']
        dt_matches = eval_img['dtMatches']  # TxD matrix
        
        # Check if any detection matches multiple GTs
        gt_matches_by_dt = defaultdict(set)
        for t_idx in range(dt_matches.shape[0]):  # For each IoU threshold
            for dt_idx in range(dt_matches.shape[1]):  # For each detection
                gt_id = dt_matches[t_idx, dt_idx]
                if gt_id > 0:  # If matched to a GT
                    gt_matches_by_dt[dt_idx].add(gt_id)
        
        # If any detection matches multiple GTs, add image to set
        for dt_matches in gt_matches_by_dt.values():
            if len(dt_matches) > 1:
                multi_match_img_ids.add(img_id)
                break
    
    return multi_match_img_ids

def filter_coco_annotations(ground_truth_path, output_path, multi_match_img_ids):
    """
    Filter COCO annotations to remove images with multi-match detections
    and their associated annotations
    """
    # Read original annotations
    with open(ground_truth_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create deep copy to modify
    filtered_data = copy.deepcopy(coco_data)
    
    # Remove images with multi-matches
    filtered_data['images'] = [
        img for img in filtered_data['images'] 
        if img['id'] not in multi_match_img_ids
    ]
    
    # Remove annotations for those images
    filtered_data['annotations'] = [
        ann for ann in filtered_data['annotations']
        if ann['image_id'] not in multi_match_img_ids
    ]
    
    # Save filtered annotations
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f)
    
    # Print statistics
    orig_img_count = len(coco_data['images'])
    orig_ann_count = len(coco_data['annotations'])
    new_img_count = len(filtered_data['images'])
    new_ann_count = len(filtered_data['annotations'])
    
    print(f"Original dataset: {orig_img_count} images, {orig_ann_count} annotations")
    print(f"Filtered dataset: {new_img_count} images, {new_ann_count} annotations")
    print(f"Removed {len(multi_match_img_ids)} images with multi-match detections")
    print(f"Removed {orig_ann_count - new_ann_count} annotations")