import json
import random
from collections import defaultdict

def reduce_annotations(input_json_path, output_json_path, num_samples=10):
    # Read the original JSON file
    with open(input_json_path, 'r') as file:
        data = json.load(file)
    
    # Randomly select num_samples images
    selected_images = random.sample(data['images'], num_samples)
    selected_image_ids = {img['id'] for img in selected_images}
    
    # Filter annotations for selected images
    filtered_annotations = [
        ann for ann in data['annotations']
        if ann['image_id'] in selected_image_ids
    ]
    
    # Get unique category IDs from filtered annotations
    used_category_ids = {ann['category_id'] for ann in filtered_annotations}
    
    # Filter categories to only include those that are used
    filtered_categories = [
        cat for cat in data['categories']
        if cat['id'] in used_category_ids
    ]
    
    # Create new reduced dataset
    reduced_data = {
        'info': data['info'],
        'licenses': data['licenses'],
        'images': selected_images,
        'annotations': filtered_annotations,
        'categories': filtered_categories
    }
    
    # Save reduced dataset
    with open(output_json_path, 'w') as file:
        json.dump(reduced_data, file, indent=2)
    
    # Print statistics
    print(f"Original dataset statistics:")
    print(f"- Number of images: {len(data['images'])}")
    print(f"- Number of annotations: {len(data['annotations'])}")
    print(f"- Number of categories: {len(data['categories'])}")
    print(f"\nReduced dataset statistics:")
    print(f"- Number of images: {len(selected_images)}")
    print(f"- Number of annotations: {len(filtered_annotations)}")
    print(f"- Number of categories: {len(filtered_categories)}")

# Usage example
input_path = "/home/emirhan/datasets/object_detection/coco/annotations/instances_val2017_original.json"
output_path = "/home/emirhan/datasets/object_detection/coco/annotations/instances_val2017.json"
reduce_annotations(input_path, output_path, num_samples=10)