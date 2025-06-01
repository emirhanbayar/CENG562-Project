import matplotlib.pyplot as plt
import numpy as np

# Data preparation
metrics = [
    'AP', 'AP50', 'AP75', 'APsmall', 'APmedium', 'APlarge',
    'AR1', 'AR10', 'AR100', 'ARsmall', 'ARmedium', 'ARlarge'
]

DINO  = [0.491, 0.667, 0.536, 0.327, 0.524, 0.630,
         0.379, 0.651, 0.728, 0.563, 0.767, 0.883]

DINO_mini_gt_cheat = [0.838, 0.838, 0.838, 0.738, 0.850, 0.827,
                      0.535, 0.962, 0.994, 0.989, 0.994, 0.999]

#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.838
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.838
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.838
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.738
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.850
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.827
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.535
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.962
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.994
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.989
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.994
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.999

# Set up the plot
fig, ax = plt.subplots(figsize=(15, 8))

# Set width of bars and positions of the bars
bar_width = 0.25
r1 = np.arange(len(metrics))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Create bars
bars1 = plt.bar(r1, DINO, width=bar_width,
                label='DINO (6 layers, 118K images)', color='lightblue')
bars2 = plt.bar(r2, DINO_mini_gt_cheat, width=bar_width,
                label='DINO-GT-CHEAT (1 layer, 25K images)', color='darkblue')

# Function to add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', rotation=0, fontsize=6)

# Add value labels to each set of bars
add_value_labels(bars1)
add_value_labels(bars2)

# Add labels and title
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('DINO Performance vs Training Set Size')
plt.xticks([r + bar_width for r in range(len(metrics))], metrics, rotation=45)

# Add legend
plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Add grid for better readability
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.show()