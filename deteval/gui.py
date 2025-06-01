import cv2
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
import os

class DetectionVisualizer:
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.current_img_idx = 0
        self.show_tp = tk.BooleanVar(value=True)
        self.show_fp = tk.BooleanVar(value=True)
        self.show_fn = tk.BooleanVar(value=True)
        self.show_duplicates = tk.BooleanVar(value=False)
        self.duplicate_iou_thresh = 0.5
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Detection Visualizer")
        
        # Create controls frame
        controls = ttk.Frame(self.root)
        controls.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Checkboxes for toggling detection types
        ttk.Checkbutton(controls, text="Show TP", variable=self.show_tp, 
                       command=self.update_image).pack(side=tk.LEFT)
        ttk.Checkbutton(controls, text="Show FP", variable=self.show_fp,
                       command=self.update_image).pack(side=tk.LEFT)
        ttk.Checkbutton(controls, text="Show FN", variable=self.show_fn,
                       command=self.update_image).pack(side=tk.LEFT)
        ttk.Checkbutton(controls, text="Show Duplicates", variable=self.show_duplicates,
                       command=self.update_image).pack(side=tk.LEFT)
        
        # IOU threshold slider for duplicates
        ttk.Label(controls, text="Duplicate IOU Threshold:").pack(side=tk.LEFT)
        self.iou_slider = ttk.Scale(controls, from_=0.1, to=0.9, orient=tk.HORIZONTAL,
                                  length=200, value=0.5, command=self.on_iou_change)
        self.iou_slider.pack(side=tk.LEFT)
        
        # Navigation buttons
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(side=tk.TOP, fill=tk.X, padx=5)
        ttk.Button(nav_frame, text="Previous", command=self.prev_image).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="Next", command=self.next_image).pack(side=tk.LEFT)
        
        # Image display
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        # Load evaluation results
        self.load_eval_results()
        
        # Show first image
        self.update_image()
        
    def load_eval_results(self):
        """Load evaluation results from eval_vis directory"""
        self.image_files = []
        self.results = {}
        
        eval_dir = 'eval_vis'
        if not os.path.exists(eval_dir):
            print(f"Warning: {eval_dir} directory not found")
            return
            
        for f in os.listdir(eval_dir):
            if f.endswith('_debug.jpg'):
                img_id = int(f.split('_')[0])
                self.image_files.append(img_id)
                
        self.image_files.sort()
    
    def compute_iou(self, box1, box2):
        """Compute IOU between two boxes in [x, y, w, h] format"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to [x1, y1, x2, y2] format
        box1 = [x1, y1, x1 + w1, y1 + h1]
        box2 = [x2, y2, x2 + w2, y2 + h2]
        
        # Calculate intersection area
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
        
    def find_duplicates(self, detections):
        """Find duplicate detections based on IOU threshold"""
        duplicates = []
        n = len(detections)
        
        for i in range(n):
            for j in range(i + 1, n):
                if detections[i]['type'] == detections[j]['type'] == 'TP':
                    iou = self.compute_iou(detections[i]['bbox'], detections[j]['bbox'])
                    if iou > self.duplicate_iou_thresh:
                        duplicates.append((i, j))
        
        return duplicates
    
    def draw_detections(self, img, detections):
        """Draw detections on image with current visibility settings"""
        colors = {
            'TP': (0, 255, 0),    # Green
            'FP': (0, 0, 255),    # Red
            'FN': (255, 0, 0)     # Blue
        }
        
        # First draw all normal detections
        for i, det in enumerate(detections):
            if ((det['type'] == 'TP' and self.show_tp.get()) or
                (det['type'] == 'FP' and self.show_fp.get()) or
                (det['type'] == 'FN' and self.show_fn.get())):
                
                x, y, w, h = [int(b) for b in det['bbox']]
                color = colors[det['type']]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                
                if det['type'] in ['TP', 'FP']:
                    label = f"{det['type']} {det['category']} {det['score']:.2f}"
                else:
                    label = f"{det['type']} {det['category']}"
                
                cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1)
        
        # Then highlight duplicates if enabled
        if self.show_duplicates.get():
            duplicates = self.find_duplicates(detections)
            for i, j in duplicates:
                # Draw connection between duplicate boxes
                box1 = detections[i]['bbox']
                box2 = detections[j]['bbox']
                
                pt1 = (int(box1[0] + box1[2]/2), int(box1[1] + box1[3]/2))
                pt2 = (int(box2[0] + box2[2]/2), int(box2[1] + box2[3]/2))
                
                cv2.line(img, pt1, pt2, (255, 255, 0), 2)  # Yellow line
                
        return img
    
    def update_image(self):
        """Update the displayed image with current settings"""
        if not self.image_files:
            return
            
        img_id = self.image_files[self.current_img_idx]
        img_path = os.path.join(self.img_dir, f"{img_id:012d}.jpg")
        debug_path = os.path.join('eval_vis', f"{img_id:012d}_debug.jpg")
        
        # Load the original image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            return
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # TODO: Load detections from evaluation results
        # For now, we'll parse them from the debug image filename pattern
        # You should modify this to use your actual detection results
        
        # Resize image to fit canvas while maintaining aspect ratio
        height, width = img.shape[:2]
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        scale = min(canvas_width/width, canvas_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        img = cv2.resize(img, (new_width, new_height))
        
        # Convert to PhotoImage
        img_tk = ImageTk.PhotoImage(Image.fromarray(img))
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=img_tk)
        self.canvas.image = img_tk  # Keep a reference
        
    def on_iou_change(self, value):
        """Handle IOU threshold slider change"""
        self.duplicate_iou_thresh = float(value)
        self.update_image()
        
    def prev_image(self):
        """Show previous image"""
        if self.current_img_idx > 0:
            self.current_img_idx -= 1
            self.update_image()
            
    def next_image(self):
        """Show next image"""
        if self.current_img_idx < len(self.image_files) - 1:
            self.current_img_idx += 1
            self.update_image()
            
    def run(self):
        """Start the visualization tool"""
        self.root.mainloop()

# Usage example
if __name__ == "__main__":
    vis = DetectionVisualizer(img_dir="path/to/your/images")
    vis.run()