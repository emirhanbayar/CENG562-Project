import os
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from pycocotools.coco import COCO


class FlirPairedDataset(Dataset):
    """
    Dataset for paired RGB-IR images with their annotations
    """
    def __init__(self, rgb_annotation_file, ir_annotation_file, 
                 rgb_img_dir, ir_img_dir, 
                 processor=None, transform=None, split='train'):
        """
        Args:
            rgb_annotation_file (str): Path to RGB annotations JSON file
            ir_annotation_file (str): Path to IR annotations JSON file
            rgb_img_dir (str): Directory with RGB images
            ir_img_dir (str): Directory with IR images
            processor (Owlv2Processor): Processor for OWLv2 model
            transform: Custom transforms to apply
            split (str): 'train' or 'val'
        """
        self.rgb_coco = COCO(rgb_annotation_file)
        self.ir_coco = COCO(ir_annotation_file)
        self.rgb_img_dir = rgb_img_dir
        self.ir_img_dir = ir_img_dir
        self.processor = processor
        self.transform = transform
        self.split = split
        
        # Get all image IDs (use the IR dataset as reference)
        self.img_ids = list(sorted(self.ir_coco.imgs.keys()))
        
        # Get the mapping between IR and RGB image IDs
        # This is a simplified approach assuming same order and 1:1 mapping
        # In a real scenario, you might need a more robust matching logic
        self.ir_to_rgb_mapping = {ir_id: rgb_id for ir_id, rgb_id in zip(self.ir_coco.imgs.keys(), self.rgb_coco.imgs.keys())}
        
        # Get category information
        self.categories = {cat['id']: cat['name'] for cat in self.ir_coco.cats.values()}
        
        # Precompute image paths and annotation data for faster __getitem__
        self.sample_data = []
        for ir_img_id in self.img_ids:
            rgb_img_id = self.ir_to_rgb_mapping[ir_img_id]
            
            # Get image paths
            ir_img_info = self.ir_coco.imgs[ir_img_id]
            ir_img_path = os.path.join(self.ir_img_dir, ir_img_info['file_name'])
            
            rgb_img_info = self.rgb_coco.imgs[rgb_img_id]
            rgb_img_path = os.path.join(self.rgb_img_dir, rgb_img_info['file_name'])
            
            # Get annotations for IR image
            ir_ann_ids = self.ir_coco.getAnnIds(imgIds=ir_img_id)
            ir_anns = self.ir_coco.loadAnns(ir_ann_ids)
            
            # Extract bounding boxes and class labels
            boxes = []
            category_ids = []
            class_names = []
            
            for ann in ir_anns:
                # COCO format: [x_min, y_min, width, height]
                x, y, w, h = ann['bbox']
                # Convert to [x_min, y_min, x_max, y_max]
                boxes.append([x, y, x + w, y + h])
                category_ids.append(ann['category_id'])
                class_names.append(self.categories[ann['category_id']])
            
            # Store precomputed data
            self.sample_data.append({
                'ir_img_path': ir_img_path,
                'rgb_img_path': rgb_img_path,
                'boxes': boxes,
                'category_ids': category_ids,
                'class_names': class_names,
                'image_id': ir_img_id,
                'img_size': (ir_img_info['height'], ir_img_info['width'])
            })
    
    def __len__(self):
        return len(self.sample_data)
    
    def __getitem__(self, idx):
        # Get precomputed data for this sample
        sample = self.sample_data[idx]
        
        # Load IR image
        ir_image = Image.open(sample['ir_img_path']).convert("RGB")
        
        # Load RGB image
        rgb_image = Image.open(sample['rgb_img_path']).convert("RGB")
        
        # Create annotation dictionary with tensor conversion
        target = {
            'boxes': torch.tensor(sample['boxes'], dtype=torch.float32) if sample['boxes'] else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(sample['category_ids'], dtype=torch.int64) if sample['category_ids'] else torch.zeros(0, dtype=torch.int64),
            'class_names': sample['class_names'],
            'image_id': sample['image_id'],
            'ir_path': sample['ir_img_path'],
            'rgb_path': sample['rgb_img_path'],
            'img_size': sample['img_size']
        }
        
        # Apply processor if provided
        if self.processor:
            # Prepare text prompts for each class
            text_prompts = [[f"a photo of a {cls_name}"] for cls_name in sample['class_names']]
            if not text_prompts:  # If no objects, use a dummy prompt
                text_prompts = [["a photo of an object"]]
                
            # Process inputs for OWLv2
            inputs = self.processor(text=text_prompts, images=ir_image, return_tensors="pt")
            
            # Add processed RGB image
            rgb_inputs = self.processor(images=rgb_image, return_tensors="pt")
            inputs['rgb_pixel_values'] = rgb_inputs.pixel_values
            
            # Add original images and targets
            inputs['ir_image'] = ir_image
            inputs['rgb_image'] = rgb_image
            inputs['target'] = target
            
            return inputs
            
        # If no processor, return images and target
        return {
            'ir_image': ir_image, 
            'rgb_image': rgb_image, 
            'target': target
        }


def visualize_sample(ir_image, rgb_image, target, processed_ir_image, processed_rgb_image, output_path, idx):
    """
    Visualize an image sample with bounding boxes and save it, including transformed images
    
    Args:
        ir_image: Original IR image
        rgb_image: Original RGB image
        target: Ground truth annotations
        processed_ir_image: Transformed IR image (model input)
        processed_rgb_image: Transformed RGB image (model input)
        output_path: Directory to save visualizations
        idx: Sample index
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Row 1: Original images with ground truth boxes
    axes[0, 0].imshow(ir_image)
    axes[0, 0].set_title("Original IR Image (Ground Truth)")
    
    axes[0, 1].imshow(rgb_image)
    axes[0, 1].set_title("Original RGB Image (Ground Truth)")
    
    # Add ground truth bounding boxes to original images
    boxes = target['boxes'].numpy() if isinstance(target['boxes'], torch.Tensor) else target['boxes']
    class_names = target['class_names']
    
    for i, (box, cls) in enumerate(zip(boxes, class_names)):
        x1, y1, x2, y2 = box
        
        # Add to IR image
        rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        axes[0, 0].add_patch(rect)
        axes[0, 0].text(x1, y1, cls, bbox=dict(facecolor='yellow', alpha=0.5))
        
        # Add to RGB image
        rect2 = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        axes[0, 1].add_patch(rect2)
        axes[0, 1].text(x1, y1, cls, bbox=dict(facecolor='yellow', alpha=0.5))
    
    # Row 2: Transformed images (model inputs)
    
    # Process and display transformed IR image
    if processed_ir_image is not None:
        # Convert processed tensor image to numpy for visualization
        if isinstance(processed_ir_image, torch.Tensor):
            processed_np = processed_ir_image.permute(1, 2, 0).cpu().numpy()
            # Denormalize if needed
            processed_np = np.clip(processed_np, 0, 1)  # Ensure values are between 0 and 1
            processed_np = (processed_np * 255).astype(np.uint8)
        else:
            processed_np = processed_ir_image
            
        axes[1, 0].imshow(processed_np)
        axes[1, 0].set_title("Transformed IR Image (Model Input)")
    else:
        axes[1, 0].axis('off')
        axes[1, 0].set_title("Processed IR image not available")
    
    # Process and display transformed RGB image
    if processed_rgb_image is not None:
        # Convert processed tensor image to numpy for visualization
        if isinstance(processed_rgb_image, torch.Tensor):
            processed_rgb_np = processed_rgb_image.permute(1, 2, 0).cpu().numpy()
            # Denormalize if needed
            processed_rgb_np = np.clip(processed_rgb_np, 0, 1)  # Ensure values are between 0 and 1
            processed_rgb_np = (processed_rgb_np * 255).astype(np.uint8)
        else:
            processed_rgb_np = processed_rgb_image
            
        axes[1, 1].imshow(processed_rgb_np)
        axes[1, 1].set_title("Transformed RGB Image (Model Input)")
    else:
        # Display information about the image if RGB transformation not available
        axes[1, 1].axis('off')
        info_text = (
            f"Image ID: {target['image_id']}\n"
            f"Original Size: {target['img_size']}\n"
            f"Number of Objects: {len(target['class_names'])}\n"
            f"Classes: {', '.join(target['class_names'])}"
        )
        axes[1, 1].text(0.1, 0.5, info_text, fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.8))
        axes[1, 1].set_title("Image Information")
    
    # Save figure
    fig.tight_layout()
    plt.savefig(os.path.join(output_path, f"sample_{idx}.png"))
    plt.close(fig)