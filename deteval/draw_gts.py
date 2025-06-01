import json
import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

def visualize_coco_annotations(ground_truth_path, img_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize COCO api for ground truth annotations
    coco_gt = COCO(ground_truth_path)
    
    # Get all image ids
    img_ids = coco_gt.getImgIds()
    
    # Generate random colors for each category
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    num_categories = len(categories)
    colors = np.random.randint(0, 255, size=(num_categories, 3), dtype=np.uint8)
    
    # Create a dictionary to map category IDs to their index (for color assignment)
    cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(categories)}
    
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
        
        # Draw each annotation
        for ann in anns:
            # Get category color
            color = colors[cat_id_to_idx[ann['category_id']]].tolist()
            
            
            # Draw bounding box
            if 'bbox' in ann:
                x, y, w, h = [int(coord) for coord in ann['bbox']]
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
                
                # Add category name
                cat_info = coco_gt.loadCats([ann['category_id']])[0]
                cat_name = cat_info['name']
                cv2.putText(
                    vis_img,
                    cat_name,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
        
        # Save the visualized image
        output_path = os.path.join(output_dir, img_info['file_name'])
        cv2.imwrite(output_path, vis_img)

if __name__ == "__main__":
    # Your provided paths
    ground_truth_path = '/home/emirhan/datasets/object_detection/coco/annotations/instances_train2017.json'
    img_dir = '/home/emirhan/datasets/object_detection/coco/train2017'
    output_dir = 'gts_visualized'
    
    visualize_coco_annotations(ground_truth_path, img_dir, output_dir)
    print("Visualization complete! Check the output directory for results.")