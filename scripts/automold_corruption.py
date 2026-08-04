import os
import sys
import cv2
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "Automold--Road-Augmentation-Library"))
import Automold as am

# Gets the project directory's name
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import OUTPUT_DIR, REGION, FILES

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_BASE_DIR = ROOT_DIR / OUTPUT_DIR
OUTPUT_FOG_DIR = ROOT_DIR / "data/sets/nuscenes-automold/Fog"
OUTPUT_SNOW_DIR = ROOT_DIR / "data/sets/nuscenes-automold/Snow"

TARGET_FOLDERS = ["samples", "sweeps"]

SEVERITIES = {
    'easy': 0.3,
    'mid': 0.6,
    'hard': 0.9
}

def process_dataset():
    print("Starting sequential Automold dataset augmentation with fixed severities...")
    
    for folder in TARGET_FOLDERS:
        folder_path = INPUT_BASE_DIR / folder
        
        if not folder_path.exists():
            print(f"Warning: Directory {folder_path} does not exist. Skipping.")
            continue

        image_files = list(folder_path.rglob("*.jpg"))
        total_images = len(image_files)
        
        print(f"Found {total_images} images in {folder_path}...")

        for idx, img_path in enumerate(image_files):
            rel_path = img_path.relative_to(INPUT_BASE_DIR)
            
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

            if (idx + 1) % 500 == 0:
                print(f"Processed {idx + 1}/{total_images} images in {folder}...")

    print("Augmentation complete. Images saved to data/sets/nuscenes-automold/")

if __name__ == "__main__":
    process_dataset()
