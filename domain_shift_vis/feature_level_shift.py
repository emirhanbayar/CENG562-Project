import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import timm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import random
import cv2
from torchvision import transforms

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

def get_paired_images(ir_dir, rgb_dir, num_samples=100):
    """Get paired IR and RGB images."""
    # Get list of IR images
    ir_files = sorted(glob.glob(os.path.join(ir_dir, "*.jpeg")))
    
    # Get corresponding RGB images (might have different extensions)
    paired_images = []
    for ir_path in ir_files:
        filename = os.path.basename(ir_path)
        # Remove extension and find corresponding RGB file
        base_name = os.path.splitext(filename)[0]
        rgb_path_jpg = os.path.join(rgb_dir, f"{base_name}.jpg")
        rgb_path_jpeg = os.path.join(rgb_dir, f"{base_name}.jpeg")
        
        # Check which path exists
        if os.path.exists(rgb_path_jpg):
            paired_images.append((ir_path, rgb_path_jpg))
        elif os.path.exists(rgb_path_jpeg):
            paired_images.append((ir_path, rgb_path_jpeg))
    
    # Sample a subset if needed
    if num_samples and num_samples < len(paired_images):
        paired_images = random.sample(paired_images, num_samples)
    
    return paired_images

def load_and_prepare_model():
    """Load and prepare the SAM model for feature extraction."""
    # Load the model
    model = timm.create_model(
        'samvit_base_patch16.sa1b',
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
    )
    model = model.eval()

    # Get model specific transforms
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    
    return model, transform

def extract_features(model, transform, image_paths, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Extract features from a list of images using the model."""
    model = model.to(device)
    features = []
    
    for img_path in tqdm(image_paths, desc="Extracting features"):
        try:
            img = Image.open(img_path).convert('RGB')  # Convert to RGB (even IR images)
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                feature = model(img_tensor)
                
            # Move to CPU and convert to numpy
            features.append(feature.cpu().numpy().squeeze())
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return np.array(features)

def visualize_feature_space(ir_features, rgb_features, method='pca', save_path=None):
    """Visualize feature space using PCA or t-SNE."""
    # Combine features
    all_features = np.vstack([ir_features, rgb_features])
    
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(all_features)
        title = "PCA Visualization of Feature Space"
        explained_var = reducer.explained_variance_ratio_
        subtitle = f"Explained variance: {explained_var[0]:.3f}, {explained_var[1]:.3f}"
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42)
        embedding = reducer.fit_transform(all_features)
        title = "t-SNE Visualization of Feature Space"
        subtitle = ""
    
    # Split back to IR and RGB
    ir_embedding = embedding[:len(ir_features)]
    rgb_embedding = embedding[len(ir_features):]
    
    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(ir_embedding[:, 0], ir_embedding[:, 1], c='blue', label='IR', alpha=0.6)
    plt.scatter(rgb_embedding[:, 0], rgb_embedding[:, 1], c='red', label='RGB', alpha=0.6)
    
    plt.title(f"{title}\n{subtitle}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()

def visualize_pixel_distributions(ir_paths, rgb_paths, num_samples=5, save_path=None):
    """Visualize pixel value distributions using histograms."""
    # Sample some image pairs
    if num_samples and num_samples < len(ir_paths):
        indices = random.sample(range(len(ir_paths)), num_samples)
        sampled_ir = [ir_paths[i] for i in indices]
        sampled_rgb = [rgb_paths[i] for i in indices]
    else:
        sampled_ir = ir_paths
        sampled_rgb = rgb_paths
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    
    for i, (ir_path, rgb_path) in enumerate(zip(sampled_ir, sampled_rgb)):
        # Load images
        ir_img = np.array(Image.open(ir_path).convert('L'))  # Convert IR to grayscale
        rgb_img = np.array(Image.open(rgb_path).convert('RGB'))
        
        # Plot IR image
        axes[i, 0].imshow(ir_img, cmap='gray')
        axes[i, 0].set_title(f"IR Image {i+1}")
        axes[i, 0].axis('off')
        
        # Plot RGB image
        axes[i, 1].imshow(rgb_img)
        axes[i, 1].set_title(f"RGB Image {i+1}")
        axes[i, 1].axis('off')
        
        # Plot histograms
        axes[i, 2].hist(ir_img.ravel(), bins=256, alpha=0.5, color='blue', label='IR')
        
        # For RGB, plot histogram for each channel
        colors = ['red', 'green', 'blue']
        for j, color in enumerate(colors):
            axes[i, 2].hist(rgb_img[:,:,j].ravel(), bins=256, alpha=0.5, color=color, label=f'RGB {color}')
        
        axes[i, 2].set_title(f"Pixel Distributions")
        axes[i, 2].legend()
        axes[i, 2].set_xlim([0, 255])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved pixel distributions to {save_path}")
    
    plt.show()

def visualize_aggregate_histograms(ir_paths, rgb_paths, num_samples=100, save_path=None):
    """Visualize aggregate pixel histograms across multiple images."""
    # Sample some image pairs
    if num_samples and num_samples < len(ir_paths):
        indices = random.sample(range(len(ir_paths)), num_samples)
        sampled_ir = [ir_paths[i] for i in indices]
        sampled_rgb = [rgb_paths[i] for i in indices]
    else:
        sampled_ir = ir_paths
        sampled_rgb = rgb_paths
    
    # Initialize histogram arrays
    ir_hist = np.zeros(256)
    r_hist = np.zeros(256)
    g_hist = np.zeros(256)
    b_hist = np.zeros(256)
    
    # Accumulate histograms
    for ir_path, rgb_path in tqdm(zip(sampled_ir, sampled_rgb), desc="Calculating histograms"):
        # Load images
        ir_img = np.array(Image.open(ir_path).convert('L'))  # Convert IR to grayscale
        rgb_img = np.array(Image.open(rgb_path).convert('RGB'))
        
        # Compute histograms
        ir_h, _ = np.histogram(ir_img.ravel(), bins=256, range=[0, 256])
        r_h, _ = np.histogram(rgb_img[:,:,0].ravel(), bins=256, range=[0, 256])
        g_h, _ = np.histogram(rgb_img[:,:,1].ravel(), bins=256, range=[0, 256])
        b_h, _ = np.histogram(rgb_img[:,:,2].ravel(), bins=256, range=[0, 256])
        
        # Accumulate
        ir_hist += ir_h
        r_hist += r_h
        g_hist += g_h
        b_hist += b_h
    
    # Normalize
    ir_hist = ir_hist / ir_hist.sum()
    r_hist = r_hist / r_hist.sum()
    g_hist = g_hist / g_hist.sum()
    b_hist = b_hist / b_hist.sum()
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(ir_hist, color='black', alpha=0.7, label='IR')
    plt.plot(r_hist, color='red', alpha=0.7, label='RGB (Red)')
    plt.plot(g_hist, color='green', alpha=0.7, label='RGB (Green)')
    plt.plot(b_hist, color='blue', alpha=0.7, label='RGB (Blue)')
    
    plt.title(f"Aggregate Pixel Value Distributions ({num_samples} images)")
    plt.xlabel("Pixel Value")
    plt.ylabel("Normalized Frequency")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved aggregate histograms to {save_path}")
    
    plt.show()

def main():
    # Define paths
    base_dir = "/home/emirhan/datasets/object_detection/FLIR/"
    ir_train_dir = os.path.join(base_dir, "FLIR_ADAS_IR/train2017")
    rgb_train_dir = os.path.join(base_dir, "FLIR_ADAS_RGB/train2017")
    
    # Get paired images
    print("Finding paired images...")
    paired_images = get_paired_images(ir_train_dir, rgb_train_dir, num_samples=200)
    print(f"Found {len(paired_images)} paired images")
    
    # Separate IR and RGB paths
    ir_paths = [pair[0] for pair in paired_images]
    rgb_paths = [pair[1] for pair in paired_images]
    
    # Visualize pixel distributions
    print("Visualizing individual pixel distributions...")
    visualize_pixel_distributions(ir_paths, rgb_paths, num_samples=5, 
                                 save_path="pixel_distributions.png")
    
    print("Visualizing aggregate pixel histograms...")
    visualize_aggregate_histograms(ir_paths, rgb_paths, num_samples=100,
                                  save_path="aggregate_histograms.png")
    
    # Load model
    print("Loading model...")
    model, transform = load_and_prepare_model()
    
    # Extract features
    print("Extracting features from IR images...")
    ir_features = extract_features(model, transform, ir_paths[:100])
    
    print("Extracting features from RGB images...")
    rgb_features = extract_features(model, transform, rgb_paths[:100])
    
    # Visualize feature space
    print("Visualizing feature space with PCA...")
    visualize_feature_space(ir_features, rgb_features, method='pca',
                           save_path="pca_visualization.png")
    
    print("Visualizing feature space with t-SNE...")
    visualize_feature_space(ir_features, rgb_features, method='tsne',
                           save_path="tsne_visualization.png")
    
    print("Done!")

if __name__ == "__main__":
    main()