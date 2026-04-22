from dotenv import load_dotenv
from openxlab.dataset import get

load_dotenv()

print("Starting OpenXLab download for nuScenes-C...")
get(dataset_repo='OpenDataLab/nuScenes-C', target_path='data/sets/')
print("Download Complete")
