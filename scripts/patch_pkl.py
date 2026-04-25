import sys
import os
import pickle

from pathlib import Path
# Gets the project directory's name
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import CLEAN_PKL, OUTPUT_PKL_PATHS

from pathlib import Path
from constants import CLEAN_PKL, OUTPUT_PKL_PATHS

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
with open(CLEAN_PKL, 'rb') as f:
    clean_data = pickle.load(f)

for output_path in OUTPUT_PKL_PATHS:
    p = Path(output_path)
    severity = p.parent.name
    weather = p.parent.parent.name

    target_str = f'./data/nuScenes-c/{weather}/{severity}'
    print(f"\nPatching metadata for {weather} ({severity})...")

    patched_data = replace_paths(clean_data, target_str)

    p.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(patched_data, f)

    print(f"Success! Saved to: {output_path}")

print("All pkl files have been generated!")
