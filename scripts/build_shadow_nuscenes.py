"""
Researcher(s): Hasan Zahid, Nadman Abdullah Bin Faisal, Vaibhav Puram

Artifact: 
nuScenes-C Relative Shadow Directory Builder
Methodology: Design Science Research (Cycle II: Solution Design)

Purpose:
This utility constructs "shadow directories" for the corrupted nuScenes-C dataset 
using relative symbolic links. It restructures the extracted corrupted camera 
data into the required `samples/` hierarchy and symlinks the unaltered metadata 
and LiDAR sensor data from the clean nuScenes dataset. This ensures seamless 
compatibility with BEVFormer and BEVFusion data loaders for post-hoc OOD 
robustness benchmarking while preventing massive data duplication.

This ensures seamless compatibility with BEVFormer and BEVFusion data loaders 
for post-hoc OOD robustness benchmarking while preventing massive data duplication.

NOTE: This file explicitly excludes symlinking to corrupted LIDAR_TOP data 
(data/sets/nuscenes-c/nuScenes-C/samples/*), as it is not required for the 
BEVFormer architecture's processing pipeline.
"""

import os
import shutil

from constants import CORE_NUSCENES_FOLDER, LIDAR_SENSORS

print("Building Shadow Directories for BEVFormer (Relative Paths)...")

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

clean_dir = os.path.join(root_dir, "data", "sets", "nuscenes")
corrupt_dir = os.path.join(root_dir, "data", "sets", "nuscenes-c", "nuScenes-c")

for corruption in os.listdir(corrupt_dir):
    corruption_path = os.path.join(corrupt_dir, corruption)
    if not os.path.isdir(corruption_path): continue

    for severity in os.listdir(corruption_path):
        severity_path = os.path.join(corruption_path, severity)
        if not os.path.isdir(severity_path): continue
        
        print(f"Formatting: {corruption} -> {severity}")
        
        samples_dir = os.path.join(severity_path, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        for item in os.listdir(severity_path):
            if item.startswith("CAM_") and os.path.isdir(os.path.join(severity_path, item)):
                shutil.move(os.path.join(severity_path, item), os.path.join(samples_dir, item))

        for folder in CORE_NUSCENES_FOLDER:
            target = os.path.join(clean_dir, folder)
            link = os.path.join(severity_path, folder)
            
            rel_target = os.path.relpath(target, start=severity_path)
            
            if os.path.islink(link):
                os.unlink(link)
                
            if not os.path.exists(link):
                os.symlink(rel_target, link)

        for sensor in LIDAR_SENSORS:
            target = os.path.join(clean_dir, "samples", sensor)
            link = os.path.join(samples_dir, sensor)
            
            rel_target = os.path.relpath(target, start=samples_dir)
            
            if os.path.islink(link):
                os.unlink(link)
                
            if not os.path.exists(link):
                os.symlink(rel_target, link)

print("\n\ttRelative shadow directories complete.")
