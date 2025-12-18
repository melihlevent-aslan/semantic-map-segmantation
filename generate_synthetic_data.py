import numpy as np
import cv2
import os
import random

"""
Synthetic Map Generator for Semantic Segmentation
-------------------------------------------------
Author: Melih Levent Aslan
Description: 
    Generates synthetic training data (Images + Masks) mimicking 
    historical map features (Roads, Buildings).
    
    Classes:
    0: Background (White)
    1: Roads (Grey)
    2: Buildings (Red)
"""

# --- CONFIGURATION ---
NUM_SAMPLES = 20        # Number of image-mask pairs to generate
IMG_SIZE = 512          # Resolution (512x512)
OUTPUT_DIR = "data"     # Root directory for data

# Define Colors (BGR format for OpenCV)
COLOR_BACKGROUND = (255, 255, 255)  # White
COLOR_ROAD = (100, 100, 100)        # Grey
COLOR_BUILDING = (0, 0, 200)        # Red

# Define Class IDs for Mask
CLASS_BG = 0
CLASS_ROAD = 1
CLASS_BUILDING = 2

def setup_directories():
    """Creates necessary directories if they don't exist."""
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)

def create_synthetic_map(index):
    """
    Generates a single synthetic map sample and its corresponding mask.
    Args:
        index (int): The index of the sample for naming.
    """
    # 1. Initialize Canvas
    image = np.full((IMG_SIZE, IMG_SIZE, 3), COLOR_BACKGROUND, dtype=np.uint8)
    mask = np.full((IMG_SIZE, IMG_SIZE), CLASS_BG, dtype=np.uint8)

    # 2. Draw Roads (Random Polylines)
    # We draw roads first so buildings can overlap them (occlusion simulation)
    num_roads = random.randint(3, 8)
    for _ in range(num_roads):
        pt1 = (random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE))
        pt2 = (random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE))
        thickness = random.randint(10, 25)
        
        # Draw on Image
        cv2.line(image, pt1, pt2, COLOR_ROAD, thickness)
        # Draw on Mask
        cv2.line(mask, pt1, pt2, CLASS_ROAD, thickness)

    # 3. Draw Buildings (Random Rectangles)
    num_buildings = random.randint(5, 15)
    for _ in range(num_buildings):
        x = random.randint(0, IMG_SIZE - 60)
        y = random.randint(0, IMG_SIZE - 60)
        w = random.randint(30, 80)
        h = random.randint(30, 80)
        
        # Draw on Image
        cv2.rectangle(image, (x, y), (x+w, y+h), COLOR_BUILDING, -1)
        # Draw on Mask
        cv2.rectangle(mask, (x, y), (x+w, y+h), CLASS_BUILDING, -1)

    # 4. Save to Disk
    img_filename = os.path.join(OUTPUT_DIR, "images", f"map_{index:03d}.png")
    mask_filename = os.path.join(OUTPUT_DIR, "masks", f"mask_{index:03d}.png")
    
    cv2.imwrite(img_filename, image)
    
    # For visualization, we scale the mask values (0,1,2 -> 0,100,200) 
    # so they are visible when opened as PNG.
    # In real training, you would save raw (0,1,2).
    cv2.imwrite(mask_filename, mask * 100) 

    print(f"[INFO] Generated sample: {img_filename}")

if __name__ == "__main__":
    print(f"🚀 Starting Synthetic Data Generation ({NUM_SAMPLES} samples)...")
    setup_directories()
    
    for i in range(NUM_SAMPLES):
        create_synthetic_map(i)
        
    print("✅ Data generation complete. Check the 'data/' folder.")