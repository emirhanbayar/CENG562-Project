import json

def convert_annotations_to_predictions(annotations_file, output_file):
    """
    Convert COCO annotation format to prediction format.
    
    Args:
        annotations_file (str): Path to the input annotations JSON file
        output_file (str): Path to save the converted predictions JSON file
    """
    # Read the annotations file
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # Extract only the annotations part and convert format
    predictions = []
    
    for idx, ann in enumerate(data['annotations']):
        prediction = {
            'id': idx,  # Create a new sequential ID
            'image_id': ann['image_id'],
            'category_id': ann['category_id'],
            'bbox': ann['bbox'],
            'area': ann['area'],
            'score': 1.0  # Since these are ground truth annotations, set confidence to 1.0
        }
        predictions.append(prediction)
    
    # Save the converted format
    with open(output_file, 'w') as f:
        json.dump({'predictions': predictions}, f, indent=2)

    print(f"Converted {len(predictions)} annotations to predictions format")
    return predictions

def validate_conversion(original_file, converted_file):
    """
    Validate that the conversion maintained all essential information.
    
    Args:
        original_file (str): Path to the original annotations file
        converted_file (str): Path to the converted predictions file
    """
    # Read both files
    with open(original_file, 'r') as f:
        original = json.load(f)
    with open(converted_file, 'r') as f:
        converted = json.load(f)
    
    # Basic validation
    original_count = len(original['annotations'])
    converted_count = len(converted['predictions'])
    
    print("\nValidation Results:")
    print(f"Original annotations count: {original_count}")
    print(f"Converted predictions count: {converted_count}")
    
    if original_count == converted_count:
        print("✓ Count match successful")
    else:
        print("⨯ Count mismatch!")
    
    # Sample check of the first item
    if original_count > 0 and converted_count > 0:
        orig = original['annotations'][0]
        conv = converted['predictions'][0]
        
        print("\nFirst item comparison:")
        print(f"Original bbox: {orig['bbox']}")
        print(f"Converted bbox: {conv['bbox']}")
        print(f"Original category_id: {orig['category_id']}")
        print(f"Converted category_id: {conv['category_id']}")

if __name__ == "__main__":
    # File paths
    annotations_file = "/home/emirhan/datasets/object_detection/coco/annotations/instances_val2017.json"
    output_file = "converted_predictions.json"
    
    # Perform conversion
    predictions = convert_annotations_to_predictions(annotations_file, output_file)
    
    # Validate the conversion
    validate_conversion(annotations_file, output_file)