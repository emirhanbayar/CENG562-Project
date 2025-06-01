import json
from pycocotools.coco import COCO
import copy

def generate_predictions(annotation_file, output_file):
    """
    Generate predictions by copying each ground truth box 80 times,
    with original category having 80% confidence and remaining 20% 
    distributed among other categories.
    """
    # Initialize COCO api for annotations
    coco = COCO(annotation_file)
    
    # Get all image ids
    img_ids = coco.getImgIds()
    
    # Initialize predictions list
    predictions = []
    
    # List of valid COCO category IDs (excluding N/A categories)
    valid_categories = [i for i in range(1, 91) 
                       if i not in [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]]
    
    # Number of valid categories excluding the original one
    num_other_categories = len(valid_categories) - 1
    
    # Process each image
    for img_id in img_ids:
        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Process each annotation
        for ann in anns:
            # Get the original bbox and category
            bbox = ann['bbox']
            original_category = ann['category_id']
            
            # Calculate confidence for other categories
            
            # Create predictions for all categories
            for category_id in valid_categories:
                # Create prediction entry
                prediction = {
                    'image_id': img_id,
                    'category_id': category_id,
                    'bbox': copy.deepcopy(bbox),
                    'score': 0.90 if category_id == original_category else 0.80
                }
                
                predictions.append(prediction)
    
    # Create the final predictions dictionary
    predictions_dict = {
        'predictions': predictions
    }
    
    # Save predictions to file
    with open(output_file, 'w') as f:
        json.dump(predictions_dict, f)

if __name__ == "__main__":
    # Paths
    annotation_file = '/home/emirhan/datasets/object_detection/coco/annotations/instances_val2017.json'
    output_file = 'predictions_80_categories.json'
    
    # Generate predictions
    generate_predictions(annotation_file, output_file)
    print(f"Generated predictions saved to {output_file}")