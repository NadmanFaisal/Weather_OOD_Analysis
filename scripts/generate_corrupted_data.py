# src: https://github.com/bethgelab/imagecorruptions/blob/master/test_demo.py
# scr: https://www.nuscenes.org/tutorials/nuscenes_tutorial.html
"""
Researcher(s): Hasan Zahid, Nadman Abdullah Bin Faisal, Vaibhav Puram

Artifact: 
nuScenes Data Ingestion & Integrity Pipeline
Methodology: Design Science Research (Cycle II: Solution Design)

Purpose:
This pipeline acts as our synthetic weather engine. It 
ingests clean camera keyframes from the nuScenes-mini 
dataset and mathematically corrupts them to simulate adverse 
driving conditions.

Note: 
This uses the standard imagecorruptions library (Michaelis et al., 2019) 
to generate 2D prototype corruptions for safety monitor validation.
"""

import os
import sys
import cv2
import glob

from tqdm import tqdm
from imagecorruptions import corrupt
from nuscenes.nuscenes import NuScenes

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import OUTPUT_DIR, CORRUPTION_MAP, NUSCENES_CAMERAS

# Where the corrupted images are saved to
BASE_CORRUPTED_DIR = os.path.join(os.path.dirname(os.path.normpath(OUTPUT_DIR)), "nuscenes_corrupted")


def apply_prototype_corruption(image_array, corruption_type, severity=3):
    """
    Applies the mathematical weather corruption to a single image array.
    """

    # Swaps BGR to RGB
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Apply coppurtion maths with specific corruptions
    imagenet_c_name = CORRUPTION_MAP[corruption_type]
    corrupted_rgb = corrupt(image_rgb, corruption_name=imagenet_c_name, severity=severity)
    
    # Swaps RBG back to BGR
    corrupted_bgr = cv2.cvtColor(corrupted_rgb, cv2.COLOR_RGB2BGR) # type: ignore
    return corrupted_bgr

def generate_weather_split_physical(target_weather, cameras):
    """
    Physically corrupt ALL images (both samples and sweeps) 
    required for temporal models.
    """
    weather_dir = os.path.join(BASE_CORRUPTED_DIR, target_weather)
    print(f"\n[ {target_weather.upper()} ] Scaffolding directory: {weather_dir}")

    all_image_paths = []
    for cam in cameras:
        samples_path = os.path.join(OUTPUT_DIR, 'samples', cam, '*.jpg')
        all_image_paths.extend(glob.glob(samples_path))
        
        sweeps_path = os.path.join(OUTPUT_DIR, 'sweeps', cam, '*.jpg')
        all_image_paths.extend(glob.glob(sweeps_path))

    if not all_image_paths:
        print(f"Warning: No images found for {target_weather}. Check OUTPUT_DIR.")
        return

    progress_bar = tqdm(
        total=len(all_image_paths), 
        desc=f"Generating {target_weather} (Samples + Sweeps)", 
        ascii=True, 
        dynamic_ncols=True
    )

    for img_path in all_image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            corrupted_img = apply_prototype_corruption(img, corruption_type=target_weather, severity=3)
            
            relative_path = os.path.relpath(img_path, OUTPUT_DIR)
            save_path = os.path.join(weather_dir, relative_path)
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, corrupted_img)
            
        progress_bar.update(1)

    progress_bar.close()

def main():
    print("Starting Physical Data Corruption Pipeline...")
    
    for target_weather in CORRUPTION_MAP.keys():
        generate_weather_split_physical(target_weather, NUSCENES_CAMERAS)

    print("\n\tAll nuScenes-mini-C Testbeds (Samples + Sweeps) Generated Successfully!")

if __name__ == "__main__":
    main()
