"""
Researcher(s): Hasan Zahid, Nadman Abdullah Bin Faisal, Vaibhav Puram

Artifact: 
nuScenes-C Annotation (PKL) Metadata Patcher
Methodology: Design Science Research (Cycle II: Solution Design)

Purpose:
This utility dynamically patches the clean nuScenes annotation metadata 
(.pkl files) to redirect image file paths to the newly generated 
nuscenes-automold shadow directories. By recursively replacing base data paths 
for each specific corruption type and severity level, it ensures that 
BEVFormer and BEVFusion dataloaders fetch the correct Out-of-Distribution 
(OOD) camera data while preserving the original dataset's bounding box 
and LiDAR metadata intact.
"""

import sys
import os
import pickle

from pathlib import Path
# Gets the project directory's name
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import CLEAN_PKL, AUTOMOLD_OUTPUT_PKL_PATHS

def replace_paths(obj, target_str):
    if isinstance(obj, dict):
        return {k: replace_paths(v, target_str) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_paths(v, target_str) for v in obj]
    elif isinstance(obj, str):
        # Mimic the BEVFormer's required path
        return obj.replace('./data/nuscenes', target_str)
    else:
        return obj

print("Loading clean annotations...")

try:
    with open(CLEAN_PKL, 'rb') as f:
        clean_data = pickle.load(f)
except FileNotFoundError:
    print(f"\tCRITICAL ERROR: Could not find the clean pkl file at {CLEAN_PKL}")
    print("Please check your path. Aborting script.")
    sys.exit(1)
except Exception as e:
    print(f"\tCRITICAL ERROR: Failed to load {CLEAN_PKL}. Reason: {e}")
    sys.exit(1)

success_count = 0
failed_paths = []

for output_path in AUTOMOLD_OUTPUT_PKL_PATHS:
    try:
        p = Path(output_path)
        severity = p.parent.name
        weather = p.parent.parent.name
        
        # The target str that is replaced into the pkl file to point to the right files
        target_str = f'./data/nuscenes_automold/{weather}/{severity}'
        print(f"\nPatching metadata for {weather} ({severity})...")

        patched_data = replace_paths(clean_data, target_str)

        p.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(patched_data, f)

        print(f"Success! Saved to: {output_path}")
        success_count += 1

    except Exception as e:
        print(f"\tERROR processing {output_path}: {e}")
        failed_paths.append((output_path, str(e)))

print("\n" + "="*40)
print("\tBATCH PROCESSING SUMMARY")
print("="*40)
print(f"Total Attempted: {len(AUTOMOLD_OUTPUT_PKL_PATHS)}")
print(f"Successful:      {success_count}")
print(f"Failed:          {len(failed_paths)}")

if failed_paths:
    print("\nFAILED FILES:")
    for path, error in failed_paths:
        print(f"\t- {path}\n   Reason: {error}")
else:
    print("\nAll pkl files have been generated flawlessly!")
