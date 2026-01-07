import os
import zipfile
import av
from tqdm import tqdm
from PIL import Image

TOTAL_FRAME_PER_VIDEO = 600

class NvidiaDatasetExtractor:
    def __init__(self, raw_root, output_root):
        self.raw_root = raw_root
        self.output_root = output_root
        self.camera_root = os.path.join(raw_root, "camera")

    def get_available_cameras(self):
        if not os.path.exists(self.camera_root):
            return []
        cameras = [d for d in os.listdir(self.camera_root) 
                   if os.path.isdir(os.path.join(self.camera_root, d))]
        return sorted(cameras)

    def extract_chunk(self, chunk_id, target_cameras, max_frames=5):
        chunk_str = f"{chunk_id:04d}"
        
        total_video_frames = TOTAL_FRAME_PER_VIDEO
        interval = int(total_video_frames / max_frames)
        
        print(f"\n  Processing Chunk {chunk_id} for {len(target_cameras)} cameras...")

        results = {}

        for cam_name in target_cameras:
            zip_filename = f"{cam_name}.chunk_{chunk_str}.zip"
            zip_path = os.path.join(self.camera_root, cam_name, zip_filename)
            
            if not os.path.exists(zip_path):
                print(f"    Skipping {cam_name}: Zip not found.")
                results[cam_name] = False
                continue

            chunk_output_dir = os.path.join(self.output_root, f"chunk_{chunk_str}_{max_frames}frames", cam_name)
            os.makedirs(chunk_output_dir, exist_ok=True)

            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    all_files = z.namelist()
                    mp4_files = [f for f in all_files if f.endswith(".mp4")]
                    
                    if not mp4_files:
                        results[cam_name] = False
                        continue

                    for video_name in tqdm(mp4_files, desc=f"   {cam_name}", leave=False):
                        self._process_single_video(z, video_name, chunk_output_dir, interval, max_frames)
                
                results[cam_name] = True

            except Exception as e:
                print(f"   Error in {cam_name}: {e}")
                results[cam_name] = False

        return results

    def _process_single_video(self, zip_ref, video_name, output_dir, interval, max_limit):
        clip_id = video_name.split('.')[0]
        
        try:
            with zip_ref.open(video_name) as video_file:
                container = av.open(video_file)
                stream = container.streams.video[0]
                
                saved_count = 0
                for i, frame in enumerate(container.decode(stream)):
                    if i % interval == 0:
                        self._save_frame(frame, clip_id, i, output_dir)
                        saved_count += 1
                    
                    if saved_count >= max_limit:
                        break
        except Exception:
            pass

    def _save_frame(self, frame, clip_id, frame_num, output_dir):
        save_name = f"{clip_id}_frame{frame_num:04d}.jpg"
        save_path = os.path.join(output_dir, save_name)
        img = frame.to_image()
        img.save(save_path, quality=80)
