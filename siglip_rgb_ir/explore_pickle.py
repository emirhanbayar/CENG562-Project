import pickle

# Load the features
with open('/arf/scratch/ebayar/CENG562-Project/siglip_rgb_ir/siglip_results_largest_model/extracted_features_train.pkl', 'rb') as f:
    features_data = pickle.load(f)

# Access hook results (patch embeddings before attention pooling)
rgb_patch_embeddings = features_data['rgb_patch_embeddings']  # [N, 256, 768]
print(f"RGB Patch Embeddings Shape: {rgb_patch_embeddings.shape}")
ir_patch_embeddings = features_data['ir_patch_embeddings']    # [N, 256, 768]

# Also available:
rgb_final_features = features_data['rgb_features']            # [N, 768] - after pooling
ir_final_features = features_data['ir_features']              # [N, 768] - after pooling
metadata = features_data['metadata']                          # List of image info