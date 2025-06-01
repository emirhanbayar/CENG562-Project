import os
import argparse
import torch
import shutil
import json
from torch.utils.data import DataLoader
from transformers import Owlv2Processor

# Import dataset and model modules
from datasets import FlirPairedDataset, visualize_sample
from models import ZeroShotDetector, Owlv2RGBIRInvariant

from torch.optim import AdamW
from tqdm import tqdm # For progress bar

def parse_args():
    parser = argparse.ArgumentParser(description="zero-shot OWLv2 model for RGB-IR object detection")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/owlv2-base-patch16-finetuned",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference",
    )
    # Dataset arguments
    parser.add_argument(
        "--ir_train_annotations",
        type=str,
        default="/home/emirhan/datasets/object_detection/FLIR_ADAS_IR/annotations/instances_train2017.json",
        help="Path to IR train annotations JSON file",
    )
    parser.add_argument(
        "--ir_val_annotations",
        type=str,
        default="/home/emirhan/datasets/object_detection/FLIR_ADAS_IR/annotations/instances_val2017.json",
        help="Path to IR validation annotations JSON file",
    )
    parser.add_argument(
        "--rgb_train_annotations",
        type=str,
        default="/home/emirhan/datasets/object_detection/FLIR_ADAS_RGB/annotations/instances_train2017.json",
        help="Path to RGB train annotations JSON file",
    )
    parser.add_argument(
        "--rgb_val_annotations",
        type=str,
        default="/home/emirhan/datasets/object_detection/FLIR_ADAS_RGB/annotations/instances_val2017.json",
        help="Path to RGB validation annotations JSON file",
    )
    parser.add_argument(
        "--ir_train_dir",
        type=str,
        default="/home/emirhan/datasets/object_detection/FLIR_ADAS_IR/train2017",
        help="Directory with IR train images",
    )
    parser.add_argument(
        "--ir_val_dir",
        type=str,
        default="/home/emirhan/datasets/object_detection/FLIR_ADAS_IR/val2017",
        help="Directory with IR validation images",
    )
    parser.add_argument(
        "--rgb_train_dir",
        type=str,
        default="/home/emirhan/datasets/object_detection/FLIR_ADAS_RGB/train2017",
        help="Directory with RGB train images",
    )
    parser.add_argument(
        "--rgb_val_dir",
        type=str,
        default="/home/emirhan/datasets/object_detection/FLIR_ADAS_RGB/val2017",
        help="Directory with RGB validation images",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for dataloader",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results and predictions",
    )
    parser.add_argument(
        "--debug_dir",
        type=str,
        default="debug_visualization",
        help="Directory to save debug visualizations",
    )
    parser.add_argument(
        "--num_debug_images",
        type=int,
        default=100,
        help="Number of image pairs to save during debug",
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Save visualizations of model predictions",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["debug", "zero_shot", "domain_invariant"],
        default="debug",
        help="Mode to run: debug or zero_shot",
    )

    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--save_every", type=int, default=1, help="Save model checkpoint every N epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")

    return parser.parse_args()


def extract_categories_from_coco(json_path):
    """Extract category information from COCO annotation file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['categories']


def debug_dataloading(args):
    """Test dataloader and visualize samples"""
    print("Initializing OWLv2 model and processor...")
    processor = Owlv2Processor.from_pretrained(args.model_name)
    
    print("Creating dataset and dataloader...")
    # Create dataset
    train_dataset = FlirPairedDataset(
        rgb_annotation_file=args.rgb_train_annotations,
        ir_annotation_file=args.ir_train_annotations,
        rgb_img_dir=args.rgb_train_dir,
        ir_img_dir=args.ir_train_dir,
        processor=processor,
        split='train'
    )
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: x  # Don't collate as the processor already batches
    )
    
    # Create debug directory
    if os.path.exists(args.debug_dir):
        shutil.rmtree(args.debug_dir)
    os.makedirs(args.debug_dir)
    
    print(f"Visualizing {args.num_debug_images} samples to {args.debug_dir}...")
    # Iterate through some samples and visualize them
    for i, batch in enumerate(train_dataloader):
        if i >= (args.num_debug_images // args.batch_size) + 1:
            break
            
        for j, sample in enumerate(batch):
            if i * args.batch_size + j >= args.num_debug_images:
                break
                
            # Get images and target
            ir_image = sample['ir_image']
            rgb_image = sample['rgb_image']
            target = sample['target']
            
            # Get the processed/transformed images
            processed_ir_image = sample['pixel_values'][0].cpu()
            processed_rgb_image = sample['rgb_pixel_values'][0].cpu()
            
            # Visualize and save
            idx = i * args.batch_size + j
            visualize_sample(
                ir_image,
                rgb_image,
                target,
                processed_ir_image,
                processed_rgb_image,
                args.debug_dir,
                idx
            )
            
            print(f"Saved visualization for sample {idx}")
    
    print(f"Visualization complete. Check {args.debug_dir} for results.")
    print("Dataset size:", len(train_dataset))
    print("Sample structure:", list(batch[0].keys()))


def run_zero_shot_evaluation(args):
    """Run zero-shot detection and evaluation on both RGB and IR modalities"""
    print("Initializing OWLv2 processor...")
    processor = Owlv2Processor.from_pretrained(args.model_name)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get category information
    categories = extract_categories_from_coco(args.ir_val_annotations)
    
    print(f"Found {len(categories)} categories: {[cat['name'] for cat in categories]}")
    
    # Create validation dataset
    print("Creating validation dataset...")
    val_dataset = FlirPairedDataset(
        rgb_annotation_file=args.rgb_val_annotations,
        ir_annotation_file=args.ir_val_annotations,
        rgb_img_dir=args.rgb_val_dir,
        ir_img_dir=args.ir_val_dir,
        processor=processor,
        split='val'
    )
    
    # Create dataloader with a custom collate function that preserves batch structure
    def custom_collate(batch):
        return batch
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate
    )
    
    # Initialize zero-shot detector
    print("Initializing zero-shot detector...")
    detector = ZeroShotDetector(
        model_name=args.model_name,
        device=args.device,
        categories=categories,
        score_threshold=0.3,
        save_visualizations=args.save_visualizations,
        visualization_dir=os.path.join(args.output_dir, "visualizations")
    )
    
    # Run evaluation on both RGB and IR
    print("Running zero-shot evaluation on both RGB and IR...")
    
    # Run evaluation with comparison of modalities
    metrics = detector.run_zero_shot_evaluation(
        val_dataloader,
        args.ir_val_annotations,
        args.output_dir,
        compare_modalities=True
    )
    
    # Print comparison
    print("\nModality Comparison:")
    print(f"IR AP50: {metrics['IR']['AP50']:.4f} vs RGB AP50: {metrics['RGB']['AP50']:.4f}")
    
    print(f"Results saved to {args.output_dir}")
    print(f"Metrics comparison saved to {os.path.join(args.output_dir, 'modality_comparison.csv')}")


def run_domain_invariant_learning(args):
    """Run domain invariant learning"""
    print("Initializing OWLv2 processor...")
    processor = Owlv2Processor.from_pretrained(args.model_name)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get category information (can be used for logging or evaluation later)
    # categories = extract_categories_from_coco(args.ir_train_annotations)
    # print(f"Found {len(categories)} categories: {[cat['name'] for cat in categories]}")

    # Create training dataset
    print("Creating training dataset...")
    train_dataset = FlirPairedDataset(
        rgb_annotation_file=args.rgb_train_annotations,
        ir_annotation_file=args.ir_train_annotations,
        rgb_img_dir=args.rgb_train_dir,
        ir_img_dir=args.ir_train_dir,
        processor=processor, # Pass the processor to handle text and image processing
        split='train'
    )

    # Custom collate function to handle the list of dicts from the dataset
    # and prepare a batch for the model
    def custom_collate_for_training(batch_list):
        # batch_list is a list of dicts, where each dict is an output of FlirPairedDataset.__getitem__
        # Each dict contains: 'pixel_values' (for IR), 'input_ids', 'attention_mask', 'rgb_pixel_values', 'target'
        
        # Stack pixel values and RGB pixel values
        ir_pixel_values = torch.cat([sample['pixel_values'] for sample in batch_list], dim=0)
        rgb_pixel_values = torch.cat([sample['rgb_pixel_values'] for sample in batch_list], dim=0)
        
        # For input_ids and attention_mask, they are usually padded to the max length *in that batch* by the processor.
        # If the processor in FlirPairedDataset already creates them as (1, num_queries, seq_len) or (num_queries, seq_len),
        # we need to stack them correctly.
        # The FlirPairedDataset's processor call:
        # inputs = self.processor(text=text_prompts, images=ir_image, return_tensors="pt")
        # This will make 'input_ids' of shape (num_prompts_for_this_sample, seq_length)
        # We need to ensure all samples in a batch have the same number of prompts or pad them.
        # For simplicity, assume text_prompts are handled such that they can be stacked.
        # Or, more robustly, pad input_ids and attention_masks per batch here.
        
        # Assuming text_prompts leads to input_ids of shape [num_prompts, seq_len] for each sample
        # And all samples in batch have the same num_prompts (e.g. by padding in dataset or using fixed num_prompts)
        # Let's say dataset ensures num_prompts is fixed (e.g. by taking first N, or padding prompts)
        if batch_list[0]['input_ids'] is not None:
            input_ids = torch.cat([sample['input_ids'] for sample in batch_list], dim=0) # Becomes (B * num_prompts, seq_len)
            attention_mask = torch.cat([sample['attention_mask'] for sample in batch_list], dim=0) # (B * num_prompts, seq_len)
        else: # No text prompts
            input_ids = None
            attention_mask = None
            
        targets = [sample['target'] for sample in batch_list] # List of target dicts

        return {
            'ir_pixel_values': ir_pixel_values,
            'rgb_pixel_values': rgb_pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'targets': targets
        }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 4, # Default if not in args
        collate_fn=custom_collate_for_training # Use the custom collate function
    )

    # Initialize the domain invariant model
    print(f"Initializing Owlv2RGBIRInvariant model with base {args.model_name}...")
    model = Owlv2RGBIRInvariant(
        model_name=args.model_name,
        device=args.device
    )
    model.to(args.device)

    # Optimizer
    # You might want to use different learning rates for different parts of the model
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate if hasattr(args, 'learning_rate') else 1e-5)

    num_epochs = args.num_epochs if hasattr(args, 'num_epochs') else 10 # Add num_epochs to argparse

    print(f"Starting training for {num_epochs} epochs...")
    model.train() # Set model to training mode

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_domain_loss = 0.0
        epoch_detection_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            ir_pixel_values = batch['ir_pixel_values'].to(args.device)
            rgb_pixel_values = batch['rgb_pixel_values'].to(args.device)
            input_ids = batch['input_ids'].to(args.device) if batch['input_ids'] is not None else None
            attention_mask = batch['attention_mask'].to(args.device) if batch['attention_mask'] is not None else None
            targets = batch['targets'] # List of dicts, handle on device within model or loss function

            # Forward pass
            output = model(
                ir_pixel_values=ir_pixel_values,
                rgb_pixel_values=rgb_pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                targets=targets # Pass targets for detection loss calculation
            )

            loss = output["total_loss"]
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_domain_loss += output["domain_invariance_loss"].item()
            if isinstance(output["detection_loss_ir"], torch.Tensor): # Check if it's a tensor
                 epoch_detection_loss += output["detection_loss_ir"].item()


            progress_bar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Avg Epoch Loss": f"{epoch_loss / (batch_idx + 1):.4f}"
            })

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        avg_domain_loss = epoch_domain_loss / len(train_dataloader)
        avg_detection_loss = epoch_detection_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Summary: Avg Total Loss: {avg_epoch_loss:.4f}, "
              f"Avg Domain Loss: {avg_domain_loss:.4f}, Avg Detection Loss: {avg_detection_loss:.4f}")

        # Optional: Save model checkpoint
        if (epoch + 1) % (args.save_every if hasattr(args, 'save_every') else 1) == 0: # Add save_every to argparse
            checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("Training complete.")
    # Final model save
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "debug":
        debug_dataloading(args)
    elif args.mode == "zero_shot":
        run_zero_shot_evaluation(args)
    elif args.mode == "domain_invariant":
        run_domain_invariant_learning(args)
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    args = parse_args()
    main(args)