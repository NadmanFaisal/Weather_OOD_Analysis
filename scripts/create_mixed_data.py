import pickle
import random
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import CLEAN_PKL, CORRUPTED_PKL, OUTPUT_PKL

def mix_datasets(clean_pkl_path, corrupted_pkl_path, output_pkl_path, sample_size=100):
    """
    Creates a mixed dataset containing an equal number of clean and corrupted scenes.
    """
    print("Loading Clean Data...")
    with open(clean_pkl_path, 'rb') as f:
        clean_data = pickle.load(f)
        
    print("Loading Corrupted Data (nuScenes-C)...")
    with open(corrupted_pkl_path, 'rb') as f:
        corrupted_data = pickle.load(f)

    clean_infos = clean_data['infos']
    corrupted_infos = corrupted_data['infos']
    
    max_available = min(len(clean_infos), len(corrupted_infos))
    if sample_size > max_available:
        print(f"Warning: Requested {sample_size} but only {max_available} available. Adjusting...")
        sample_size = max_available

    random.seed(42) 
    sampled_clean = random.sample(clean_infos, sample_size)
    sampled_corrupted = random.sample(corrupted_infos, sample_size)

    for item in sampled_clean:
        item['is_ood'] = 0  # 0 = Clean / In-Distribution
        
    for item in sampled_corrupted:
        item['is_ood'] = 1  # 1 = Corrupted / Out-of-Distribution

    mixed_infos = sampled_clean + sampled_corrupted
    random.shuffle(mixed_infos)

    # Package it into the OpenMMLab format
    mixed_dataset = {
        'metadata': clean_data['metadata'],
        'infos': mixed_infos
    }

    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)

    with open(output_pkl_path, 'wb') as f:
        pickle.dump(mixed_dataset, f)
        
    print(f"\nSuccess! Created mixed dataset at {output_pkl_path}")
    print(f"Total Scenes: {len(mixed_infos)} ({sample_size} Clean, {sample_size} Corrupted)")

if __name__ == '__main__':

    # We mix 50 clean and 50 corrupted to make a 100-scene test set
    mix_datasets(CLEAN_PKL, CORRUPTED_PKL, OUTPUT_PKL, sample_size=50)
