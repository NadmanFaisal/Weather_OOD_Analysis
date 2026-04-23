import os
import subprocess

from dotenv import load_dotenv
from openxlab.dataset import get

load_dotenv()

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

base_dir = os.path.join(root_dir, "data", "sets")
target_dir = os.path.join(base_dir, "nuscenes-c")

image_archive = os.path.join(base_dir, "OpenDataLab___nuScenes-C", "raw", "image", "nuScenes-C.tar.gz")
pointcloud_archive = os.path.join(base_dir, "OpenDataLab___nuScenes-C", "raw", "pointcloud", "nuScenes-C.zip")

# Download nuscenes_c from OpenxLab
print("Starting OpenXLab download for nuScenes-C...")
get(dataset_repo='OpenDataLab/nuScenes-C', target_path='data/sets/')
print("Download Complete")

print("\nStarting Extraction")
os.makedirs(target_dir, exist_ok=True)
print(f"Target directory prepared at: {target_dir}")

print(f"\nExtracting images from: {image_archive}")
print("This may take a while. Please wait...")
tar_command = [
    "tar", "-xzmf", image_archive, 
    "-C", target_dir, 
    "--no-same-owner" 
]
subprocess.run(tar_command, check=True)
print("Image extraction complete!")

print(f"\nExtracting pointclouds from: {pointcloud_archive}")
print("This may take a while. Please wait...")
zip_command = [
    "unzip", "-q", "-o", pointcloud_archive, 
    "-d", target_dir
]
subprocess.run(zip_command, check=True)
print("Pointcloud extraction complete!")

print("\nEntire nuScenes-C dataset is unpacked and ready!")
