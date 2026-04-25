import pickle

from pathlib import Path
from constants import CLEAN_PKL

clean_pkl = 'data/sets/nuscenes/nuscenes_infos_temporal_val.pkl'
fog_pkl = 'data/sets/nuscenes-c/nuScenes-c/Fog/easy/nuscenes_infos_temporal_val.pkl'

def replace_paths(obj):
    if isinstance(obj, dict):
        return {k: replace_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_paths(v) for v in obj]
    elif isinstance(obj, str):
        # Mimic the BEVFormer's required path
        return obj.replace('./data/nuscenes', './data/nuScenes-c/Fog/easy')
    else:
        return obj

print("Loading clean annotations...")
with open(clean_pkl, 'rb') as f:
    data = pickle.load(f)

print("Patching image paths for Fog...")
patched_data = replace_paths(data)

Path(fog_pkl).parent.mkdir(parents=True, exist_ok=True)

print("Saving patched annotations...")
with open(fog_pkl, 'wb') as f:
    pickle.dump(patched_data, f)

print(f"Success! Patched file saved to: {fog_pkl}")
