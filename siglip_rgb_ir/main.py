import os
import argparse
import torch
import json
import pickle
import numpy as np
import open_clip
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Import dataset from existing codebase
from datasets import FlirPairedDataset, visualize_sample


def parse_args():
    parser = argparse.ArgumentParser(description="SigLIP-based RGB-IR feature extraction and comparison")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-SO400M-14-SigLIP2",
        help="SigLIP model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--ir_train_annotations",
        type=str,
        default="/arf/scratch/ebayar/datasets/FLIR_ADAS_IR/annotations/instances_train2017.json",
        help="Path to IR train annotations JSON file",
    )
    parser.add_argument(
        "--ir_val_annotations",
        type=str,
        default="/arf/scratch/ebayar/datasets/FLIR_ADAS_IR/annotations/instances_val2017.json",
        help="Path to IR validation annotations JSON file",
    )
    parser.add_argument(
        "--rgb_train_annotations",
        type=str,
        default="/arf/scratch/ebayar/datasets/FLIR_ADAS_RGB/annotations/instances_train2017.json",
        help="Path to RGB train annotations JSON file",
    )
    parser.add_argument(
        "--rgb_val_annotations",
        type=str,
        default="/arf/scratch/ebayar/datasets/FLIR_ADAS_RGB/annotations/instances_val2017.json",
        help="Path to RGB validation annotations JSON file",
    )
    parser.add_argument(
        "--ir_train_dir",
        type=str,
        default="/arf/scratch/ebayar/datasets/FLIR_ADAS_IR/train2017",
        help="Directory with IR train images",
    )
    parser.add_argument(
        "--ir_val_dir",
        type=str,
        default="/arf/scratch/ebayar/datasets/FLIR_ADAS_IR/val2017",
        help="Directory with IR validation images",
    )
    parser.add_argument(
        "--rgb_train_dir",
        type=str,
        default="/arf/scratch/ebayar/datasets/FLIR_ADAS_RGB/train2017",
        help="Directory with RGB train images",
    )
    parser.add_argument(
        "--rgb_val_dir",
        type=str,
        default="/arf/scratch/ebayar/datasets/FLIR_ADAS_RGB/val2017",
        help="Directory with RGB validation images",
    )
    
    # Processing arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="siglip_results_largest_model",
        help="Directory to save results",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["extract_features", "compare_modalities", "debug"],
        default="extract_features",
        help="Mode to run",
    )
    parser.add_argument(
        "--num_debug_images",
        type=int,
        default=50,
        help="Number of images for debug visualization",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--extract_patch_embeddings",
        action="store_true",
        help="Extract patch embeddings before attention pooling",
    )
    parser.add_argument(
        "--no_tokenizer",
        action="store_true",
        help="Skip tokenizer loading (for image-only extraction)",
    )
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        help="Clear HuggingFace cache before starting",
    )
    
    return parser.parse_args()


def extract_categories_from_coco(json_path):
    """Extract category information from COCO annotation file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['categories']


class SigLIPFeatureExtractor:
    def __init__(self, model_name, device, no_tokenizer=False):
        self.model_name = model_name
        self.device = device
        self.no_tokenizer = no_tokenizer
        
        # Load SigLIP model
        print(f"Loading SigLIP model: {model_name}")
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained='webli'
        )
        self.model.eval()
        self.model.to(device)
        
        # Initialize tokenizer as None (only load when needed)
        self.tokenizer = None
        if no_tokenizer:
            print("Tokenizer loading disabled")
            self.tokenizer = "disabled"
        
        # Storage for intermediate features
        self.intermediate_features = {}
        self.hooks = []
        
        # Register hook to capture features before attention pooling
        self._register_hooks()
        
    def _load_tokenizer_if_needed(self):
        """Load tokenizer only when needed for text processing"""
        if self.tokenizer is None:
            try:
                print(f"Loading tokenizer for {self.model_name}...")
                self.tokenizer = open_clip.get_tokenizer(self.model_name)
                print("Tokenizer loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load tokenizer: {e}")
                print("Text feature extraction will not be available")
                self.tokenizer = "failed"  # Mark as failed to avoid repeated attempts
        
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features"""
        # Hook into the norm layer (after encoder blocks, before attn_pool)
        def capture_pre_pool_features(module, input, output):
            self.intermediate_features['pre_pool'] = output.detach()
        
        # Get the visual encoder (trunk)
        visual_model = self.model.visual
        if hasattr(visual_model, 'trunk'):
            # For models with trunk structure
            trunk = visual_model.trunk
            if hasattr(trunk, 'norm'):
                hook = trunk.norm.register_forward_hook(capture_pre_pool_features)
                self.hooks.append(hook)
                print("Registered hook on trunk.norm layer")
        else:
            # For direct visual models
            if hasattr(visual_model, 'norm'):
                hook = visual_model.norm.register_forward_hook(capture_pre_pool_features)
                self.hooks.append(hook)
                print("Registered hook on visual.norm layer")
    
    def extract_image_features(self, images, return_intermediate=False):
        """Extract features from a batch of images"""
        # Clear previous intermediate features
        self.intermediate_features.clear()
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(images.to(self.device))
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        
        if return_intermediate and 'pre_pool' in self.intermediate_features:
            # Return both final features and intermediate patch embeddings
            patch_embeddings = self.intermediate_features['pre_pool'].cpu().numpy()
            return image_features.cpu().numpy(), patch_embeddings
        else:
            return image_features.cpu().numpy()
    
    def extract_text_features(self, texts):
        """Extract features from text descriptions"""
        if self.no_tokenizer or self.tokenizer == "disabled":
            raise RuntimeError("Tokenizer disabled. Cannot extract text features.")
            
        self._load_tokenizer_if_needed()
        
        if self.tokenizer is None or self.tokenizer == "failed":
            raise RuntimeError("Tokenizer not available. Cannot extract text features.")
        
        text_tokens = self.tokenizer(texts)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.model.encode_text(text_tokens.to(self.device))
            text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu().numpy()
    
    def cleanup_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def simple_collate_fn(batch):
    """Simple collate function that doesn't use processor"""
    return batch


def clear_hf_cache():
    """Clear HuggingFace cache directory"""
    import shutil
    cache_dir = os.path.expanduser('~/.cache/huggingface')
    if os.path.exists(cache_dir):
        print(f"Clearing HuggingFace cache at {cache_dir}")
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        print("Cache cleared successfully")
    else:
        print("Cache directory does not exist")


def setup_slurm_environment():
    """Setup environment variables for better Slurm compatibility"""
    # Set environment variables for better Slurm compatibility
    cache_dir = os.path.expanduser('~/.cache/huggingface')
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'transformers')
    os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, 'datasets')
    
    # Disable tokenizers parallelism which can cause issues in multiprocessing
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Create cache directories if they don't exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)
    
    print(f"Using HuggingFace cache directory: {cache_dir}")
    print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID', 'Not running on SLURM')}")
    print(f"SLURM_PROCID: {os.environ.get('SLURM_PROCID', 'Not running on SLURM')}")


def extract_features_mode(args):
    """Extract and save features from both train and val RGB and IR images"""
    print("Starting feature extraction mode...")
    
    # Setup environment for Slurm
    setup_slurm_environment()
    
    # Clear cache if requested
    if args.clear_cache:
        clear_hf_cache()
    
    # Initialize SigLIP
    try:
        feature_extractor = SigLIPFeatureExtractor(
            args.model_name, 
            args.device, 
            no_tokenizer=args.no_tokenizer
        )
    except Exception as e:
        print(f"Error initializing SigLIP model: {e}")
        if not args.clear_cache:
            print("Attempting to clear cache and retry...")
            clear_hf_cache()
            print("Cache cleared, retrying...")
            feature_extractor = SigLIPFeatureExtractor(
                args.model_name, 
                args.device, 
                no_tokenizer=args.no_tokenizer
            )
        else:
            raise e
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process both train and val splits
    splits_data = {}
    
    for split in ['train', 'val']:
        print(f"\n{'='*50}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*50}")
        
        # Select appropriate annotation files and directories
        if split == 'train':
            rgb_annotations = args.rgb_train_annotations
            ir_annotations = args.ir_train_annotations
            rgb_img_dir = args.rgb_train_dir
            ir_img_dir = args.ir_train_dir
        else:  # val
            rgb_annotations = args.rgb_val_annotations
            ir_annotations = args.ir_val_annotations
            rgb_img_dir = args.rgb_val_dir
            ir_img_dir = args.ir_val_dir
        
        # Create dataset without processor (we'll handle transforms manually)
        dataset = FlirPairedDataset(
            rgb_annotation_file=rgb_annotations,
            ir_annotation_file=ir_annotations,
            rgb_img_dir=rgb_img_dir,
            ir_img_dir=ir_img_dir,
            processor=None,  # No processor, we'll use SigLIP transforms
            split=split
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=simple_collate_fn
        )
        
        # Storage for features
        rgb_features = []
        ir_features = []
        rgb_patch_embeddings = []
        ir_patch_embeddings = []
        image_metadata = []
        
        print(f"Extracting features from {len(dataset)} {split} image pairs...")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Extracting {split} features")):
            # Process RGB images
            rgb_images = []
            ir_images = []
            batch_metadata = []
            
            for sample in batch:
                # Apply SigLIP preprocessing
                rgb_processed = feature_extractor.preprocess_val(sample['rgb_image'])
                ir_processed = feature_extractor.preprocess_val(sample['ir_image'])
                
                rgb_images.append(rgb_processed)
                ir_images.append(ir_processed)
                batch_metadata.append({
                    'image_id': sample['target']['image_id'],
                    'class_names': sample['target']['class_names'],
                    'boxes': sample['target']['boxes'].numpy() if len(sample['target']['boxes']) > 0 else [],
                    'rgb_path': sample['target']['rgb_path'],
                    'ir_path': sample['target']['ir_path'],
                    'split': split
                })
            
            # Stack images into batches
            rgb_batch = torch.stack(rgb_images)
            ir_batch = torch.stack(ir_images)
            
            # Extract features (both final and intermediate if requested)
            if args.extract_patch_embeddings:
                rgb_feat, rgb_patches = feature_extractor.extract_image_features(rgb_batch, return_intermediate=True)
                ir_feat, ir_patches = feature_extractor.extract_image_features(ir_batch, return_intermediate=True)
                
                # Store results
                rgb_features.extend(rgb_feat)
                ir_features.extend(ir_feat)
                rgb_patch_embeddings.extend(rgb_patches)
                ir_patch_embeddings.extend(ir_patches)
            else:
                # Extract only final features
                rgb_feat = feature_extractor.extract_image_features(rgb_batch, return_intermediate=False)
                ir_feat = feature_extractor.extract_image_features(ir_batch, return_intermediate=False)
                
                # Store results
                rgb_features.extend(rgb_feat)
                ir_features.extend(ir_feat)
            
            image_metadata.extend(batch_metadata)
        
        # Convert to numpy arrays
        rgb_features = np.array(rgb_features)
        ir_features = np.array(ir_features)
        
        print(f"Extracted {split} features:")
        print(f"  RGB final features shape: {rgb_features.shape}")
        print(f"  IR final features shape: {ir_features.shape}")
        
        # Store split data
        split_data = {
            'rgb_features': rgb_features,
            'ir_features': ir_features,
            'metadata': image_metadata,
            'model_name': args.model_name,
            'split': split
        }
        
        if args.extract_patch_embeddings:
            rgb_patch_embeddings = np.array(rgb_patch_embeddings)
            ir_patch_embeddings = np.array(ir_patch_embeddings)
            print(f"  RGB patch embeddings shape: {rgb_patch_embeddings.shape}")
            print(f"  IR patch embeddings shape: {ir_patch_embeddings.shape}")
            
            split_data['rgb_patch_embeddings'] = rgb_patch_embeddings
            split_data['ir_patch_embeddings'] = ir_patch_embeddings
        
        splits_data[split] = split_data
    
    # Save combined data with both splits
    combined_features_data = {
        'train': splits_data['train'],
        'val': splits_data['val'],
        'model_name': args.model_name,
        'extraction_args': vars(args)
    }
    
    features_path = os.path.join(args.output_dir, 'extracted_features_train_val.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(combined_features_data, f)
    
    print(f"\n{'='*50}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*50}")
    print(f"Combined features saved to {features_path}")
    print(f"Train samples: {len(splits_data['train']['metadata'])}")
    print(f"Val samples: {len(splits_data['val']['metadata'])}")
    
    # Also save individual split files for compatibility
    for split, data in splits_data.items():
        split_path = os.path.join(args.output_dir, f'extracted_features_{split}.pkl')
        with open(split_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"{split.capitalize()} features also saved to {split_path}")
    
    # Cleanup hooks
    feature_extractor.cleanup_hooks()
    
    return combined_features_data


def compare_modalities_mode(args):
    """Compare RGB and IR modalities using extracted features"""
    print("Starting modality comparison mode...")
    
    # Setup environment for Slurm
    setup_slurm_environment()
    
    features_path = os.path.join(args.output_dir, 'extracted_features_train_val.pkl')
    
    if not os.path.exists(features_path):
        print(f"Combined features file not found at {features_path}. Running feature extraction first...")
        features_data = extract_features_mode(args)
    else:
        print(f"Loading features from {features_path}")
        with open(features_path, 'rb') as f:
            features_data = pickle.load(f)
    
    # Use validation split for comparison
    if 'val' in features_data:
        val_data = features_data['val']
    else:
        print("No validation data found, using all available data...")
        val_data = features_data
    
    rgb_features = val_data['rgb_features']
    ir_features = val_data['ir_features']
    metadata = val_data['metadata']
    
    # Check if patch embeddings are available
    has_patch_embeddings = 'rgb_patch_embeddings' in val_data and 'ir_patch_embeddings' in val_data
    
    if has_patch_embeddings:
        rgb_patch_embeddings = val_data['rgb_patch_embeddings']
        ir_patch_embeddings = val_data['ir_patch_embeddings']
        print(f"Found patch embeddings: RGB {rgb_patch_embeddings.shape}, IR {ir_patch_embeddings.shape}")
    
    # Compute cosine similarities between paired RGB-IR images (final features)
    print("Computing RGB-IR similarities for final features...")
    pair_similarities = []
    for i in range(len(rgb_features)):
        sim = cosine_similarity([rgb_features[i]], [ir_features[i]])[0][0]
        pair_similarities.append(sim)
    
    pair_similarities = np.array(pair_similarities)
    
    # Compute cross-modal similarities (RGB vs all IR)
    print("Computing cross-modal similarity matrix...")
    cross_modal_sim = cosine_similarity(rgb_features, ir_features)
    
    # If patch embeddings are available, compute patch-level similarities
    patch_similarities = None
    if has_patch_embeddings:
        print("Computing patch-level similarities...")
        patch_similarities = []
        for i in range(len(rgb_patch_embeddings)):
            # Average similarity across all patch pairs
            rgb_patches = rgb_patch_embeddings[i]  # Shape: (num_patches, dim)
            ir_patches = ir_patch_embeddings[i]    # Shape: (num_patches, dim)
            patch_sim_matrix = cosine_similarity(rgb_patches, ir_patches)
            # Take mean of diagonal (corresponding patches) or max similarity
            avg_patch_sim = np.mean(np.diag(patch_sim_matrix))
            max_patch_sim = np.max(patch_sim_matrix)
            patch_similarities.append({
                'avg_corresponding': avg_patch_sim,
                'max_similarity': max_patch_sim,
                'patch_sim_matrix': patch_sim_matrix
            })
    
    # Analysis
    print("\n" + "="*50)
    print("MODALITY COMPARISON RESULTS")
    print("="*50)
    print(f"Average RGB-IR pair similarity (final features): {pair_similarities.mean():.4f} ± {pair_similarities.std():.4f}")
    print(f"Median RGB-IR pair similarity: {np.median(pair_similarities):.4f}")
    print(f"Min RGB-IR pair similarity: {pair_similarities.min():.4f}")
    print(f"Max RGB-IR pair similarity: {pair_similarities.max():.4f}")
    
    if patch_similarities:
        avg_patch_sims = [p['avg_corresponding'] for p in patch_similarities]
        max_patch_sims = [p['max_similarity'] for p in patch_similarities]
        avg_patch_sims = np.array(avg_patch_sims)
        max_patch_sims = np.array(max_patch_sims)
        
        print(f"\nPatch-level similarities:")
        print(f"Average corresponding patch similarity: {avg_patch_sims.mean():.4f} ± {avg_patch_sims.std():.4f}")
        print(f"Average max patch similarity: {max_patch_sims.mean():.4f} ± {max_patch_sims.std():.4f}")
    
    # Find best and worst matches
    best_match_idx = np.argmax(pair_similarities)
    worst_match_idx = np.argmin(pair_similarities)
    
    print(f"\nBest RGB-IR match:")
    print(f"  Similarity: {pair_similarities[best_match_idx]:.4f}")
    print(f"  Image ID: {metadata[best_match_idx]['image_id']}")
    print(f"  Classes: {metadata[best_match_idx]['class_names']}")
    
    print(f"\nWorst RGB-IR match:")
    print(f"  Similarity: {pair_similarities[worst_match_idx]:.4f}")
    print(f"  Image ID: {metadata[worst_match_idx]['image_id']}")
    print(f"  Classes: {metadata[worst_match_idx]['class_names']}")
    
    # Save results
    results = {
        'pair_similarities': pair_similarities,
        'cross_modal_similarities': cross_modal_sim,
        'patch_similarities': patch_similarities,
        'statistics': {
            'mean': pair_similarities.mean(),
            'std': pair_similarities.std(),
            'median': np.median(pair_similarities),
            'min': pair_similarities.min(),
            'max': pair_similarities.max()
        },
        'best_match': {
            'index': best_match_idx,
            'similarity': pair_similarities[best_match_idx],
            'metadata': metadata[best_match_idx]
        },
        'worst_match': {
            'index': worst_match_idx,
            'similarity': pair_similarities[worst_match_idx],
            'metadata': metadata[worst_match_idx]
        }
    }
    
    if patch_similarities:
        results['patch_statistics'] = {
            'avg_corresponding_mean': avg_patch_sims.mean(),
            'avg_corresponding_std': avg_patch_sims.std(),
            'max_similarity_mean': max_patch_sims.mean(),
            'max_similarity_std': max_patch_sims.std()
        }
    
    results_path = os.path.join(args.output_dir, 'modality_comparison.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Create visualization
    fig_height = 12 if patch_similarities else 8
    plt.figure(figsize=(12, fig_height))
    
    subplot_rows = 3 if patch_similarities else 2
    
    plt.subplot(subplot_rows, 2, 1)
    plt.hist(pair_similarities, bins=50, alpha=0.7, edgecolor='black')
    plt.title('RGB-IR Pair Similarities Distribution (Final Features)')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.axvline(pair_similarities.mean(), color='red', linestyle='--', label=f'Mean: {pair_similarities.mean():.3f}')
    plt.legend()
    
    plt.subplot(subplot_rows, 2, 2)
    plt.imshow(cross_modal_sim[:50, :50], cmap='viridis')
    plt.title('Cross-Modal Similarity Matrix (50x50 subset)')
    plt.xlabel('IR Images')
    plt.ylabel('RGB Images')
    plt.colorbar()
    
    plt.subplot(subplot_rows, 2, 3)
    sorted_sims = np.sort(pair_similarities)
    plt.plot(sorted_sims)
    plt.title('Sorted RGB-IR Pair Similarities')
    plt.xlabel('Rank')
    plt.ylabel('Similarity')
    plt.grid(True)
    
    plt.subplot(subplot_rows, 2, 4)
    # Accuracy at different thresholds for retrieval
    thresholds = np.linspace(0.1, 0.9, 50)
    accuracies = []
    for thresh in thresholds:
        correct = 0
        for i in range(len(rgb_features)):
            # Find most similar IR image to RGB image i
            similarities = cross_modal_sim[i, :]
            best_match = np.argmax(similarities)
            if best_match == i and similarities[best_match] >= thresh:
                correct += 1
        accuracies.append(correct / len(rgb_features))
    
    plt.plot(thresholds, accuracies)
    plt.title('Retrieval Accuracy vs Similarity Threshold')
    plt.xlabel('Similarity Threshold')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Add patch-level analysis if available
    if patch_similarities:
        plt.subplot(subplot_rows, 2, 5)
        plt.hist(avg_patch_sims, bins=30, alpha=0.7, edgecolor='black', label='Avg Corresponding')
        plt.hist(max_patch_sims, bins=30, alpha=0.7, edgecolor='black', label='Max Similarity')
        plt.title('Patch-level Similarities Distribution')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.subplot(subplot_rows, 2, 6)
        # Show patch similarity matrix for best matching pair
        best_patch_sim = patch_similarities[best_match_idx]['patch_sim_matrix']
        plt.imshow(best_patch_sim, cmap='viridis')
        plt.title(f'Patch Similarity Matrix (Best Match - ID: {metadata[best_match_idx]["image_id"]})')
        plt.xlabel('IR Patches')
        plt.ylabel('RGB Patches')
        plt.colorbar()
    
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'modality_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to {results_path}")
    print(f"Visualization saved to {plot_path}")


def debug_mode(args):
    """Debug mode to visualize some samples"""
    print("Starting debug mode...")
    
    # Setup environment for Slurm
    setup_slurm_environment()
    
    # Clear cache if requested
    if args.clear_cache:
        clear_hf_cache()
    
    # Create dataset without processor for validation split
    val_dataset = FlirPairedDataset(
        rgb_annotation_file=args.rgb_val_annotations,
        ir_annotation_file=args.ir_val_annotations,
        rgb_img_dir=args.rgb_val_dir,
        ir_img_dir=args.ir_val_dir,
        processor=None,
        split='val'
    )
    
    # Initialize SigLIP for transforms
    feature_extractor = SigLIPFeatureExtractor(
        args.model_name, 
        args.device, 
        no_tokenizer=args.no_tokenizer
    )
    
    debug_dir = os.path.join(args.output_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    print(f"Saving {args.num_debug_images} debug visualizations to {debug_dir}")
    
    for i in range(min(args.num_debug_images, len(val_dataset))):
        sample = val_dataset[i]
        
        # Apply SigLIP preprocessing
        rgb_processed = feature_extractor.preprocess_val(sample['rgb_image'])
        ir_processed = feature_extractor.preprocess_val(sample['ir_image'])
        
        # Visualize
        visualize_sample(
            sample['ir_image'],
            sample['rgb_image'],
            sample['target'],
            ir_processed,
            rgb_processed,
            debug_dir,
            i
        )
        
        if (i + 1) % 10 == 0:
            print(f"Saved {i + 1} debug samples")
    
    print(f"Debug visualization complete. Check {debug_dir} for results.")
    
    # Cleanup hooks
    feature_extractor.cleanup_hooks()


def main():
    args = parse_args()
    
    print(f"Using device: {args.device}")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_name}")
    
    # Setup environment early for all modes
    setup_slurm_environment()
    
    if args.mode == "extract_features":
        extract_features_mode(args)
    elif args.mode == "compare_modalities":
        compare_modalities_mode(args)
    elif args.mode == "debug":
        debug_mode(args)
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()