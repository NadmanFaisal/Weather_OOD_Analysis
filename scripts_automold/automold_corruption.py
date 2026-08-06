import os
import sys
import cv2
from pathlib import Path

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

sys.path.append(os.path.join(os.path.dirname(__file__), "Automold--Road-Augmentation-Library"))
import Automold as am

# Gets the project directory's name
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import OUTPUT_DIR

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_BASE_DIR = ROOT_DIR / OUTPUT_DIR
OUTPUT_FOG_DIR = ROOT_DIR / "data/sets/nuscenes-automold/Fog"
OUTPUT_SNOW_DIR = ROOT_DIR / "data/sets/nuscenes-automold/Snow"

SEVERITIES = {
    'easy': 0.3,
    'mid': 0.6,
    'hard': 0.9
}

def get_validation_image_paths(dataroot):
    """
    Parses nuScenes metadata to return a list of relative file paths 
    (e.g., 'samples/CAM_FRONT/xxx.jpg') that belong ONLY to the validation split
    and are KEYFRAMES (samples, not sweeps).
    """
    print("Loading nuScenes metadata to filter for the validation keyframes (this takes a moment)...")
    
    # If testing with the mini split, change version to 'v1.0-mini'
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(dataroot), verbose=False)
    
    val_scenes = set(create_splits_scenes()['val'])
    
    val_image_paths = []
    
    for sd in nusc.sample_data:
        if sd['sensor_modality'] == 'camera' and sd['is_key_frame']:
            sample = nusc.get('sample', sd['sample_token'])
            scene = nusc.get('scene', sample['scene_token'])
            
            if scene['name'] in val_scenes:
                val_image_paths.append(sd['filename'])
                
    return val_image_paths

def process_dataset():
    print("Starting sequential Automold dataset augmentation for VALIDATION SAMPLES only...")
    
    val_files = get_validation_image_paths(INPUT_BASE_DIR)
    total_images = len(val_files)
    
    print(f"Found {total_images} validation sample images. Processing...")

    for idx, rel_path_str in enumerate(val_files):
        rel_path = Path(rel_path_str)
        img_path = INPUT_BASE_DIR / rel_path
        
        if not img_path.exists():
            print(f"Warning: File {img_path} not found on disk. Skipping.")
            continue
            
        all_exist = True
        for severity in SEVERITIES.keys():
            if not (OUTPUT_FOG_DIR / severity / rel_path).exists() or \
               not (OUTPUT_SNOW_DIR / severity / rel_path).exists():
                all_exist = False
                break
        
        if all_exist:
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Failed to read {img_path}")
            continue
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        for severity, coeff in SEVERITIES.items():
            fog_out_path = OUTPUT_FOG_DIR / severity / rel_path
            snow_out_path = OUTPUT_SNOW_DIR / severity / rel_path

            fog_out_path.parent.mkdir(parents=True, exist_ok=True)
            snow_out_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                foggy_images = am.add_fog([img_rgb], fog_coeff=coeff)
                snowy_images = am.add_snow([img_rgb], snow_coeff=coeff)
                
                fog_out_bgr = cv2.cvtColor(foggy_images[0], cv2.COLOR_RGB2BGR)
                snow_out_bgr = cv2.cvtColor(snowy_images[0], cv2.COLOR_RGB2BGR)

                cv2.imwrite(str(fog_out_path), fog_out_bgr)
                cv2.imwrite(str(snow_out_path), snow_out_bgr)
                
            except Exception as e:
                print(f"Error processing {img_path} at severity {severity}: {e}")

        # Print progress
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{total_images} validation images...")

    print("Augmentation complete. Images saved to data/sets/nuscenes-automold/")

if __name__ == "__main__":
    process_dataset()
