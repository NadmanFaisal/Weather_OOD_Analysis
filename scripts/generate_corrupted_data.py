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
from tqdm import tqdm
from imagecorruptions import corrupt
from nuscenes.nuscenes import NuScenes

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import OUTPUT_DIR, CORRUPTION_MAP, NUSCENES_CAMERAS, VERSIONS

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

def process_single_image(nusc, cam_token, target_weather, weather_dir):
    cam_data = nusc.get('sample_data', cam_token)
    clean_image_path = os.path.join(OUTPUT_DIR, cam_data['filename'])
    
    # Converts jgp to np array
    img = cv2.imread(clean_image_path)
    if img is None:
        return False

    corrupted_img = apply_prototype_corruption(img, corruption_type=target_weather, severity=3)

    relative_path = cam_data['filename'].replace("samples/", "")
    save_path = os.path.join(weather_dir, relative_path)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, corrupted_img)
    return True

def generate_weather_split(nusc, target_weather, cameras):
    weather_dir = os.path.join(BASE_CORRUPTED_DIR, target_weather, "samples")
    print(f"\n[ {target_weather.upper()} ] Scaffolding directory: {weather_dir}")

    # Progress bar
    total_samples = len(nusc.sample)
    progress_bar = tqdm(
        total=total_samples, 
        desc=f"Generating {target_weather}", 
        ascii=True, 
        dynamic_ncols=True
    )

    for sample in nusc.sample:
        for cam in cameras:
            cam_token = sample['data'][cam]

            # Applies corruption to the specified image
            process_single_image(nusc, cam_token, target_weather, weather_dir)
        
        progress_bar.update(1)

    progress_bar.close()
    print()

def main():
    print("Loading nuScenes v1.0-mini...")
    
    for version in VERSIONS:

        # For different versions of the data blobs, we check whether 
        # they exist. If the respective meta folder does not exist, 
        # skips creating corruption.
        valid_versions = []
        for version in VERSIONS:
            version_dir = os.path.join(OUTPUT_DIR, version)

            if os.path.exists(version_dir):
                valid_versions.append(version)
            else:
                print(f"\tFolder for '{version}' not found. It will be skipped.")

        if not valid_versions:
            print("\nNo valid datasets found on disk. Exiting pipeline.")
            return

        print(f"\nAudit passed. Proceeding with: {valid_versions}")

        for version in valid_versions:

            # Load the devkit for the specific version of data
            nusc = NuScenes(version=version, dataroot=OUTPUT_DIR, verbose=False)

            for target_weather in CORRUPTION_MAP.keys():
                generate_weather_split(nusc, target_weather, NUSCENES_CAMERAS)

    print("\tAll nuScenes-mini-C Testbeds Generated Successfully!")

if __name__ == "__main__":
    main()
