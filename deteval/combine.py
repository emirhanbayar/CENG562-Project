from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path

def combine_images(dir1, dir2, output_dir):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get directory names for headers
    dir1_name = Path(dir1).name
    dir2_name = Path(dir2).name
    
    # Get all image files from first directory
    images1 = {f.name: f for f in Path(dir1).glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']}
    
    # Process matching files from second directory
    for img2_path in Path(dir2).glob('*'):
        if img2_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
            
        if img2_path.name in images1:
            # Open both images
            img1 = Image.open(images1[img2_path.name])
            img2 = Image.open(img2_path)
            
            # Convert images to RGB if they're not
            if img1.mode != 'RGB':
                img1 = img1.convert('RGB')
            if img2.mode != 'RGB':
                img2 = img2.convert('RGB')
            
            # Resize images to same height
            height = min(img1.size[1], img2.size[1])
            ratio1 = height / img1.size[1]
            ratio2 = height / img2.size[1]
            
            new_width1 = int(img1.size[0] * ratio1)
            new_width2 = int(img2.size[0] * ratio2)
            
            img1 = img1.resize((new_width1, height), Image.Resampling.LANCZOS)
            img2 = img2.resize((new_width2, height), Image.Resampling.LANCZOS)
            
            # Create header space (50 pixels height for text)
            header_height = 50
            combined_width = new_width1 + new_width2
            combined_img = Image.new('RGB', (combined_width, height + header_height), 'white')
            
            # Add headers
            draw = ImageDraw.Draw(combined_img)
            try:
                # Try to use Arial font
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                # Fallback to default font if Arial is not available
                font = ImageFont.load_default()
            
            # Draw folder names centered above each image
            draw.text((new_width1//2, 10), dir1_name[:-6], fill='black', font=font, anchor="mm")
            draw.text((new_width1 + new_width2//2, 10), dir2_name[:-6], fill='black', font=font, anchor="mm")
            
            # Add dividing line
            draw.line([(new_width1, 0), (new_width1, header_height)], fill='black', width=2)
            
            # Paste images side by side below headers
            combined_img.paste(img1, (0, header_height))
            combined_img.paste(img2, (new_width1, header_height))
            
            # Save combined image
            output_path = os.path.join(output_dir, f"combined_{img2_path.name}")
            combined_img.save(output_path, quality=95)
            
            # Close images
            img1.close()
            img2.close()
            combined_img.close()

# Example usage
dir1 = "eval_vis"
dir2 = "eval_vis_class_agnostic"
output_dir = "class_agnostic_training_vs_class_aware_training"

combine_images(dir1, dir2, output_dir)