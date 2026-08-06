"""
Researcher(s): Hasan Zahid, Nadman Abdullah Bin Faisal, Vaibhav Puram

Artifact: 
nuScenes-C Relative Shadow Directory Builder
Methodology: Design Science Research (Cycle II: Solution Design)

Purpose:
This utility constructs "shadow directories" for the corrupted nuScenes-C dataset 
using relative symbolic links. It restructures the extracted corrupted camera 
data into the required `samples/` hierarchy and symlinks the unaltered metadata 
from the clean nuScenes dataset.

This ensures seamless compatibility with BEVFormer and BEVFusion data loaders 
for post-hoc OOD robustness benchmarking while preventing massive data duplication.

NOTE 1: This file explicitly excludes symlinking to corrupted LIDAR_TOP data 
(data/sets/nuscenes-automold/samples/*), as it is not required for the 
BEVFormer architecture's vision-only processing pipeline.

NOTE 2: This file explicitly excludes symlinking to historical sweep data (`sweeps/`) 
to prevent a temporal data leak during sequential evaluation, ensuring the model's 
recurrent memory relies strictly on corrupted OOD keyframes.
"""

import os
import shutil
import sys

# Gets the project directory's name
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import CORE_NUSCENES_FOLDER

print("Building Shadow Directories for BEVFormer (Relative Paths)...")

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

clean_dir = os.path.join(root_dir, "data", "sets", "nuscenes")
corrupt_dir = os.path.join(root_dir, "data", "sets", "nuscenes-automold")

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
            if folder == "sweeps":
                continue

            target = os.path.join(clean_dir, folder)
            link = os.path.join(severity_path, folder)
            
            rel_target = os.path.relpath(target, start=severity_path)
            
            if os.path.islink(link):
                os.unlink(link)
                
            if not os.path.exists(link):
                os.symlink(rel_target, link)

print("\n\ttRelative shadow directories complete.")
