import os
import zipfile
import av
from PIL import Image

DATASET_ROOT = os.path.abspath("./nvidia_dataset_raw")
CHUNK_ID = 1

zip_filename = f"camera_front_wide_120fov.chunk_{CHUNK_ID:04d}.zip"
zip_path = os.path.join(DATASET_ROOT, "camera", "camera_front_wide_120fov", zip_filename)

print(f"Source: {zip_path}")

try:
    with zipfile.ZipFile(zip_path, 'r') as z:
        all_files = z.namelist()
        mp4_files = [f for f in all_files if f.endswith(".mp4")]
        
        if not mp4_files:
            print("No MP4 files found.")
            exit()
            
        target_video = mp4_files[0]
        print(f"Loading Video: {target_video}")
        
        with z.open(target_video) as video_file:
            container = av.open(video_file)
            stream = container.streams.video[0]
            
            for frame in container.decode(stream):
                img = frame.to_image()
                
                output_file = "success.jpg"
                img.save(output_file)
                
                print(f"Done! Image saved to: {os.path.abspath(output_file)}")
                break 

except Exception as e:
    print(f"Error: {e}")
