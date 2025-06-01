import json
import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

COCO_NAMES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# COCO_NAMES = [
#     'N/A', 'bicycle', 'car', 'dog', 'person', 
# ]

def visualize_coco_annotations_and_predictions(ground_truth_path, predictions_path, img_dir, output_dir, score_threshold=0.5):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize COCO api for ground truth annotations
    coco_gt = COCO(ground_truth_path)
    
    # Load predictions
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    # Organize predictions by image_id for faster access
    predictions_by_image = {}
    for pred in predictions['predictions']:
        img_id = pred['image_id']
        if img_id not in predictions_by_image:
            predictions_by_image[img_id] = []
        predictions_by_image[img_id].append(pred)
    
    # Get all image ids
    img_ids = coco_gt.getImgIds()
    
    # Process each image
    for img_id in tqdm(img_ids):
        # Load image info
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Unable to read image: {img_path}")
            continue
            
        # Get annotations for this image
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        
        # Create a copy of the image for visualization
        vis_img = img.copy()
        
        # Draw ground truth annotations in RED
        for ann in anns:
            color = (0, 0, 255)  # RED for ground truth
            
            # Draw bounding box
            if 'bbox' in ann:
                x, y, w, h = [int(coord) for coord in ann['bbox']]
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 1)  # Reduced thickness
                
                # Add category name at bottom right
                cat_name = COCO_NAMES[ann['category_id']]
                label = f"GT: {cat_name}"
                if cat_name != 'N/A':
                    text_size = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    cv2.putText(
                        vis_img,
                        label,
                        (x + w - text_size[0], y + h + 10),  # bottom right
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,  # Smaller font size
                        color,
                        1    # Reduced thickness
                    )
        
        # Draw predictions in GREEN (only if score > threshold)
        if img_id in predictions_by_image:
            for pred in predictions_by_image[img_id]:
                # Only draw predictions with score higher than threshold
                if 'score' in pred and pred['score'] > score_threshold:
                    color = (0, 255, 0)  # GREEN for predictions
                    
                    # Draw bounding box
                    if 'bbox' in pred:
                        x, y, w, h = [int(coord) for coord in pred['bbox']]
                        cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 1)  # Reduced thickness
                        
                        # Add category name and score at top left
                        cat_name = COCO_NAMES[pred['category_id']]
                        if cat_name != 'N/A':
                            score = f"{pred['score']:.2f}"
                            label = f"Pred: {cat_name} {score}"
                            cv2.putText(
                                vis_img,
                                label,
                                (x, y - 5),  # top left
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,  # Smaller font size
                                color,
                                1    # Reduced thickness
                            )
        
        # Save the visualized image
        output_path = os.path.join(output_dir, img_info['file_name'])
        cv2.imwrite(output_path, vis_img)

if __name__ == "__main__":
    # Paths
    # === Configuration ===
    # Update these paths accordingly
    ground_truth_path = '/home/emirhan/datasets/object_detection/FLIR_ADAS_IR/annotations/instances_val2017_remapped.json'
    predictions_path = '/home/emirhan/DINO/logs/DINO/R50-MS4-IR/predictions.json'
    img_dir = '/home/emirhan/datasets/object_detection/FLIR_ADAS_IR/val2017/'
    output_dir = 'deneme_output_'  # Output directory for visualizations
    
    # Set score threshold here (adjust as needed)
    score_threshold = 0.1
    
    visualize_coco_annotations_and_predictions(ground_truth_path, predictions_path, img_dir, output_dir, score_threshold)
    print(f"Visualization complete! Only showing predictions with score > {score_threshold}. Check the output directory for results.")