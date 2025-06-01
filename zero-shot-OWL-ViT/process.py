import os
import json
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
import random
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from datetime import datetime
import signal
import sys
from typing import Dict, List, Optional

class ObjectDetector:
    def __init__(self, args):
        self.args = args
        self.config = self.load_config(args.config_file)
        
        # Extract categories info
        self.categories = self.config['categories']
        self.category_names = [cat['name'] for cat in self.categories]
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.categories}
        self.category_thresholds = {cat['name']: cat.get('score_threshold', 0.3) for cat in self.categories}
        
        self.processor, self.model = self.initialize_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Initialize containers for results
        self.all_annotations = []
        self.images_info = []
        self.interrupted = False
        
        # Set up signal handler for graceful interruption
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Create output directories
        os.makedirs(args.output_dir, exist_ok=True)
        if args.debug:
            self.debug_dir = os.path.join(args.output_dir, "debug_visualizations")
            os.makedirs(self.debug_dir, exist_ok=True)
            self.category_colors = None

    def load_config(self, config_file: str) -> Dict:
        """
        Load configuration from JSON file
        Expected format:
        {
            "categories": [
                {
                    "id": 1,
                    "name": "car",
                    "supercategory": "vehicle",
                    "score_threshold": 0.3
                },
                ...
            ]
        }
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
            
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        # Validate config structure
        if 'categories' not in config:
            raise ValueError("Config file must contain 'categories' key")
            
        # Validate category structure
        required_keys = ['id', 'name']
        for cat in config['categories']:
            if not all(key in cat for key in required_keys):
                raise ValueError(f"Each category must contain keys: {required_keys}")
                
        return config

    def signal_handler(self, signum, frame):
        print("\nInterrupt received. Saving current progress...")
        self.interrupted = True

    def initialize_model(self):
        """Initialize the OWL-ViT model with hardcoded name"""
        processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-finetuned")
        model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-finetuned")
        return processor, model

    def create_unique_colors(self, num_categories: int) -> List[tuple]:
        """Create a list of unique colors for visualization"""
        random.seed(42)  # For consistent colors across runs
        colors = []
        for _ in range(num_categories):
            color = (random.randint(0, 255), 
                    random.randint(0, 255), 
                    random.randint(0, 255))
            colors.append(color)
        return colors

    def draw_annotations(self, image_path: str, annotations: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on the image"""
        image = cv2.imread(image_path)
        
        # Initialize category colors if not already done
        if self.category_colors is None:
            self.category_colors = self.create_unique_colors(len(self.categories))
        
        for ann in annotations:
            category_name = self.category_id_to_name[ann['category_id']]
            if ann['score'] < self.category_thresholds[category_name]:
                continue
            
            x, y, w, h = [int(coord) for coord in ann['bbox']]
            score = ann['score']
            
            # Get color based on category ID (subtract 1 for 0-based indexing)
            color = self.category_colors[ann['category_id'] - 1]
            
            # Draw box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{category_name}: {score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            cv2.rectangle(
                image, 
                (x, y - text_height - baseline - 5), 
                (x + text_width, y), 
                color, 
                -1
            )
            
            cv2.putText(
                image, 
                label, 
                (x, y - baseline - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        return image

    def process_image(self, image_path: str, image_id: int) -> List[Dict]:
        # Load image
        image = Image.open(image_path)
        
        # Convert single-channel images to RGB
        if image.mode != 'RGB':
            # For grayscale images
            if image.mode == 'L':
                image = Image.merge('RGB', (image, image, image))
            # For other modes (including I for 32-bit integers used in some IR images)
            elif image.mode == 'I':
                # Normalize the 32-bit image to 8-bit range
                image_array = np.array(image)
                image_8bit = ((image_array - image_array.min()) * (255.0 / (image_array.max() - image_array.min()))).astype(np.uint8)
                image = Image.fromarray(image_8bit)
                image = Image.merge('RGB', (image, image, image))


        inputs = self.processor(text=self.category_names, images=image, return_tensors="pt")
        
        # Move inputs to GPU if available
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process outputs with a low initial threshold
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        min_threshold = min(self.category_thresholds.values()) * 0.5  # Use half of lowest threshold
        results = self.processor.post_process_object_detection(
            outputs=outputs, 
            threshold=min_threshold,
            target_sizes=target_sizes
        )
        
        annotations = []
        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        labels = results[0]["labels"]
        
        if boxes.nelement() > 0 and boxes.dim() > 0:
            for box, score, label in zip(boxes, scores, labels):
                # Get category name and check threshold
                category_name = self.category_names[label]
                if float(score) < self.category_thresholds[category_name]:
                    continue
                    
                box = [round(i, 2) for i in box.tolist()]
                x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
                
                # Use category ID from config instead of label index
                category_id = next(cat['id'] for cat in self.categories if cat['name'] == category_name)
                
                annotation = {
                    'area': w * h,
                    'bbox': [x, y, w, h],
                    'category_id': category_id,
                    'id': len(self.all_annotations) + len(annotations) + 1,
                    'image_id': image_id,
                    'human_annotated': "automatic",
                    'object_id': -1,
                    'out_scene': False,
                    'score': float(score)
                }
                annotations.append(annotation)
        
        return annotations

    def save_results(self):
        # Create COCO format dataset
        coco_data = {
            'info': {
                'description': 'Object Detection Results',
                'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'images': self.images_info,
            'categories': self.categories,
            'annotations': self.all_annotations
        }
        
        # Save to JSON file
        output_path = os.path.join(self.args.output_dir, "detections_coco.json")
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        print(f"Processed {len(self.images_info)} images with {len(self.all_annotations)} total detections")

    def run(self):
        for image_id, filename in enumerate(sorted(os.listdir(self.args.input_dir))):
            if self.interrupted:
                break
                
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                image_path = os.path.join(self.args.input_dir, filename)
                print(f"Processing {filename}...")
                
                try:
                    # Get image dimensions
                    with Image.open(image_path) as img:
                        width, height = img.size
                    # Add image info
                    self.images_info.append({
                        'id': image_id,
                        'file_name': filename,
                        'width': width,
                        'height': height
                    })
                    
                    # Process image and get annotations
                    image_annotations = self.process_image(
                        image_path, 
                        image_id
                    )
                    self.all_annotations.extend(image_annotations)
                    
                    # Debug visualization if requested
                    if self.args.debug and image_annotations:
                        annotated_image = self.draw_annotations(
                            image_path,
                            image_annotations
                        )
                        debug_path = os.path.join(self.debug_dir, f"debug_{filename}")
                        cv2.imwrite(debug_path, annotated_image)
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)} at line {sys.exc_info()[-1].tb_lineno}")
                    continue
        
        self.save_results()

def parse_args():
    parser = argparse.ArgumentParser(description='Flexible Object Detection with OWLv2')
    parser.add_argument('--input_dir', required=True, help='Input directory containing images')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--config_file', required=True, help='JSON configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug visualization')
    return parser.parse_args()

def main():
    args = parse_args()
    detector = ObjectDetector(args)
    detector.run()

if __name__ == "__main__":
    main()