import os
import zipfile

DATASET_ROOT = os.path.abspath("./nvidia_dataset_raw")
CHUNK_ID = 1
zip_path = os.path.join(
    DATASET_ROOT, 
    "camera", 
    "camera_front_wide_120fov", 
    f"camera_front_wide_120fov.chunk_{CHUNK_ID:04d}.zip"
)

print(f"📦 Inspecting: {os.path.basename(zip_path)}")

try:
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Get list of all files
        files = z.namelist()
        print(f"   Total files inside: {len(files)}")
        
        print("\n🔎 First 5 filenames found:")
        for f in files[:5]:
            print(f"   - {f}")
            
except Exception as e:
    print(f"❌ Error: {e}")
