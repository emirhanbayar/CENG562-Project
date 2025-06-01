import json
import matplotlib.pyplot as plt
import numpy as np

def plot_score_histogram(predictions_path, output_path, bins=50):
    # Read predictions
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    # Extract scores
    scores = [pred['score'] for pred in predictions['predictions']]
    
    # Create figure with a reasonable size
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    plt.hist(scores, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Calculate and show mean and median
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    
    # # Add vertical lines for mean and median
    # plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_score:.3f}')
    # plt.axvline(median_score, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_score:.3f}')
    
    # Customize plot
    plt.title('Distribution of Prediction Scores', pad=20)
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some statistics as text
    stats_text = f'Total Predictions: {len(scores)}\n'
    stats_text += f'Min Score: {min(scores):.3f}\n'
    stats_text += f'Max Score: {max(scores):.3f}\n'
    stats_text += f'Std Dev: {np.std(scores):.3f}'
    
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Ensure nothing is cut off
    plt.tight_layout()
    
    # Save plot
    plt.show()

if __name__ == "__main__":
    # You can analyze either the original predictions or the filtered top-N predictions
    # predictions_path = 'predictions_and_gts_visualized/predictions_top_N.json'  # or your original predictions file
    predictions_path = 'predictions.json'  # or your original predictions file
    output_path = 'score_histogram.png'
    
    plot_score_histogram(predictions_path, output_path)