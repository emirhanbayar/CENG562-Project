import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import timm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import random
from matplotlib.lines import Line2D

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

def get_paired_images(ir_dir, rgb_dir, num_samples=100, filter_pattern=None):
    """Get paired IR and RGB images."""
    # Get list of IR images
    ir_files = sorted(glob.glob(os.path.join(ir_dir, "*.jpeg")))
    
    # Filter by pattern if provided
    if filter_pattern:
        ir_files = [f for f in ir_files if filter_pattern in os.path.basename(f)]
    
    # Get corresponding RGB images
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

def extract_features_with_sam(image_paths, batch_size=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Extract features from images using SAM model."""
    # Load the model
    model = timm.create_model(
        'samvit_base_patch16.sa1b',
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
    )
    model = model.eval().to(device)
    
    # Get model specific transforms
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    
    # Process images in batches
    features = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                batch_images.append(img_tensor)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if not batch_images:
            continue
            
        # Stack batch and process
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            batch_features = model(batch_tensor)
        
        # Move to CPU and append to list
        features.append(batch_features.cpu().numpy())
    
    # Concatenate all batches
    if features:
        return np.vstack(features)
    else:
        return np.array([])

def visualize_paired_domain_shift_pca(ir_features, rgb_features, output_dir="./visualizations", 
                                     max_lines=100, line_alpha=0.3, line_color='gray'):
    """
    Visualize feature space with lines connecting paired IR and RGB features using PCA.
    
    Parameters:
    - ir_features: Features from IR images
    - rgb_features: Features from RGB images (in the same order as ir_features)
    - output_dir: Directory to save visualizations
    - max_lines: Maximum number of lines to draw for clarity
    - line_alpha: Transparency of connecting lines
    - line_color: Color of connecting lines
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure we have the same number of features for both domains
    assert len(ir_features) == len(rgb_features), "IR and RGB feature counts should match"
    
    # Combine features for dimensionality reduction
    all_features = np.vstack([ir_features, rgb_features])
    
    # Normalize features
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    
    # Split back to separate IR and RGB features after scaling
    ir_features_scaled = all_features_scaled[:len(ir_features)]
    rgb_features_scaled = all_features_scaled[len(ir_features):]
    
    # Limit the number of lines to draw for clarity
    n_lines = min(len(ir_features), max_lines)
    line_indices = random.sample(range(len(ir_features)), n_lines) if len(ir_features) > max_lines else range(len(ir_features))
    
    # Setup figure for PCA visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_features_scaled)
    
    # Split PCA results
    ir_pca = pca_result[:len(ir_features)]
    rgb_pca = pca_result[len(ir_features):]
    
    # Plot scatter points
    ax.scatter(ir_pca[:, 0], ir_pca[:, 1], c='blue', label='IR', alpha=0.6, s=30)
    ax.scatter(rgb_pca[:, 0], rgb_pca[:, 1], c='red', label='RGB', alpha=0.6, s=30)
    
    # Draw lines between corresponding points
    for idx in line_indices:
        ax.plot(
            [ir_pca[idx, 0], rgb_pca[idx, 0]], 
            [ir_pca[idx, 1], rgb_pca[idx, 1]], 
            c=line_color, alpha=line_alpha, linestyle='-', linewidth=0.8
        )
    
    # Calculate average shift vector
    avg_shift = np.mean(rgb_pca - ir_pca, axis=0)
    
    # Draw a thicker arrow for the average shift direction
    ax.arrow(
        np.mean(ir_pca[:, 0]), np.mean(ir_pca[:, 1]),
        avg_shift[0], avg_shift[1],
        head_width=0.3, head_length=0.3, fc='black', ec='black', width=0.05
    )
    
    explained_var = pca.explained_variance_ratio_
    ax.set_title(f"PCA Domain Shift Visualization (IR → RGB)\nExplained variance: {explained_var[0]:.2f}, {explained_var[1]:.2f}", fontsize=14)
    ax.set_xlabel(f"First Principal Component ({explained_var[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"Second Principal Component ({explained_var[1]:.1%} variance)", fontsize=12)
    ax.legend(fontsize=12)
    
    # Add text showing the average shift vector
    ax.text(
        0.05, 0.05, 
        f"Average shift vector: ({avg_shift[0]:.2f}, {avg_shift[1]:.2f})",
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.7),
        fontsize=12
    )
    
    # Add a custom legend for the shift lines
    custom_lines = [
        Line2D([0], [0], color=line_color, lw=1, alpha=line_alpha),
        Line2D([0], [0], color='black', lw=2)
    ]
    fig.legend(
        custom_lines, ['Individual pair shifts', 'Average shift direction'],
        loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.01), fontsize=12
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust for the bottom legend
    plt.savefig(os.path.join(output_dir, "pca_domain_shift_direction.png"), dpi=300, bbox_inches='tight')
    
    # Create another figure for shift angle analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Function to calculate angles
    def calculate_angles(source, target):
        vectors = target - source
        # Calculate angles in degrees
        angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
        # Normalize to 0-360 range
        angles = (angles + 360) % 360
        return angles
    
    # Calculate angles for PCA
    pca_angles = calculate_angles(ir_pca, rgb_pca)
    
    # Create histogram
    ax.hist(pca_angles, bins=36, alpha=0.7, color='blue')
    ax.set_title("Distribution of Domain Shift Angles (PCA)", fontsize=14)
    ax.set_xlabel("Angle (degrees)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    
    # Mark the mean angle
    mean_angle = np.mean(pca_angles)
    ax.axvline(mean_angle, color='red', linestyle='--', linewidth=2)
    ax.text(
        mean_angle + 10, ax.get_ylim()[1] * 0.9,
        f"Mean angle: {mean_angle:.1f}°",
        color='red', fontsize=12
    )
    
    # Calculate and display angular standard deviation
    angle_std = np.std(pca_angles)
    ax.text(
        0.05, 0.95,
        f"Angular std dev: {angle_std:.1f}°",
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.7),
        fontsize=12,
        verticalalignment='top'
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_domain_shift_angles.png"), dpi=300, bbox_inches='tight')
    
    # Calculate some additional statistics about the shift vectors
    shift_vectors = rgb_pca - ir_pca
    
    # Magnitude of shift vectors
    shift_magnitudes = np.linalg.norm(shift_vectors, axis=1)
    avg_magnitude = np.mean(shift_magnitudes)
    std_magnitude = np.std(shift_magnitudes)
    
    # Create another figure for shift magnitude analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(shift_magnitudes, bins=30, alpha=0.7, color='green')
    ax.axvline(avg_magnitude, color='red', linestyle='--', linewidth=2)
    ax.text(
        avg_magnitude + 0.5, ax.get_ylim()[1] * 0.9,
        f"Mean: {avg_magnitude:.2f}",
        color='red', fontsize=12
    )
    ax.set_title("Distribution of Domain Shift Magnitudes (PCA)", fontsize=14)
    ax.set_xlabel("Magnitude", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    
    # Add text showing mean and std
    ax.text(
        0.05, 0.95,
        f"Mean magnitude: {avg_magnitude:.2f}\nStd dev: {std_magnitude:.2f}",
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.7),
        fontsize=12,
        verticalalignment='top'
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_domain_shift_magnitudes.png"), dpi=300, bbox_inches='tight')
    
    print(f"PCA Domain Shift Analysis:")
    print(f"- Average shift vector: ({avg_shift[0]:.4f}, {avg_shift[1]:.4f})")
    print(f"- Mean shift angle: {mean_angle:.2f}° (std: {angle_std:.2f}°)")
    print(f"- Mean shift magnitude: {avg_magnitude:.4f} (std: {std_magnitude:.4f})")
    print(f"- PCA explained variance: {explained_var[0]:.4f}, {explained_var[1]:.4f}")
    
    return {
        'ir_pca': ir_pca,
        'rgb_pca': rgb_pca,
        'angles': pca_angles,
        'magnitudes': shift_magnitudes,
        'avg_shift': avg_shift,
        'mean_angle': mean_angle,
        'angle_std': angle_std,
        'mean_magnitude': avg_magnitude,
        'magnitude_std': std_magnitude,
        'explained_variance': explained_var
    }

def main():
    # Define paths
    base_dir = "/home/emirhan/datasets/object_detection/FLIR/"
    ir_train_dir = os.path.join(base_dir, "FLIR_ADAS_IR/train2017")
    rgb_train_dir = os.path.join(base_dir, "FLIR_ADAS_RGB/train2017")
    output_dir = os.path.join(base_dir, "domain_shift_results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get paired images
    print("Finding paired images...")
    paired_images = get_paired_images(ir_train_dir, rgb_train_dir, num_samples=100)
    print(f"Found {len(paired_images)} paired images")
    
    # Separate IR and RGB paths
    ir_paths = [pair[0] for pair in paired_images]
    rgb_paths = [pair[1] for pair in paired_images]
    
    # Extract features
    print("Extracting features from IR images...")
    ir_features = extract_features_with_sam(ir_paths)
    
    print("Extracting features from RGB images...")
    rgb_features = extract_features_with_sam(rgb_paths)
    
    # Visualize domain shift direction using PCA
    print("Visualizing domain shift direction using PCA...")
    results = visualize_paired_domain_shift_pca(ir_features, rgb_features, output_dir=output_dir)
    
    print(f"All visualizations saved to {output_dir}")
    print("Done!")

if __name__ == "__main__":
    main()