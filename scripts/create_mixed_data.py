import pickle
import random
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import CLEAN_PKL, CORRUPTED_PKL, OUTPUT_PKL

def mix_datasets(clean_pkl_path, corrupted_pkl_path, output_pkl_path):
    """
    Creates a mixed dataset containing an equal number of clean and corrupted scenes.
    """
    print("Loading Clean Data...")
    with open(clean_pkl_path, 'rb') as f:
        clean_data = pickle.load(f)
        
    print("Loading Corrupted Data (nuScenes-C)...")
    with open(corrupted_pkl_path, 'rb') as f:
        corrupted_data = pickle.load(f)

    unique_scenes_set = set()
    for info in clean_data['infos']:
        unique_scenes_set.add(info['scene_token'])
    unique_scenes = list(unique_scenes_set)
    
    unique_scenes.sort()
    
    random.seed(42) 
    random.shuffle(unique_scenes)
    midpoint = len(unique_scenes) // 2

    clean_scenes_set = set(unique_scenes[:midpoint])
    corrupted_scenes_set = set(unique_scenes[midpoint:])

    print(f"Total Scenes found: {len(unique_scenes)}")
    print(f"Allocating {len(clean_scenes_set)} scenes to Clean and {len(corrupted_scenes_set)} scenes to Foggy.")

    mixed_infos = []

    for idx in range(len(clean_data['infos'])):
        clean_info = clean_data['infos'][idx]
        corrupt_info = corrupted_data['infos'][idx]
        
        if clean_info['scene_token'] in clean_scenes_set:
            clean_info['is_ood'] = 0  # Clean
            
            for cam in clean_info['cams'].values():
                cam_tail = cam['data_path'].split('samples/')[-1]
                cam['data_path'] = './data/nuscenes/samples/' + cam_tail
                
            mixed_infos.append(clean_info)
            
        else:
            corrupt_info['is_ood'] = 1  # Corrupted
            
            for cam in corrupt_info['cams'].values():
                cam_tail = cam['data_path'].split('samples/')[-1]
                cam['data_path'] = './data/nuscenes_corrupted/fog/samples/' + cam_tail
                
            mixed_infos.append(corrupt_info)

    mixed_dataset = {
        'metadata': clean_data['metadata'],
        'infos': mixed_infos
    }

    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)

    with open(output_pkl_path, 'wb') as f:
        pickle.dump(mixed_dataset, f)
        
    print(f"\nSuccess! Created mixed scene dataset at {output_pkl_path}")
    print(f"Total Frames: {len(mixed_infos)}")

if __name__ == '__main__':

    mix_datasets(CLEAN_PKL, CORRUPTED_PKL, OUTPUT_PKL)
