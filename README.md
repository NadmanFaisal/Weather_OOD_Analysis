## How to Download Data from HuggingFace

Make sure to go through the data file structure in hugging face to understand what files to download. 

```
hf download nvidia/PhysicalAI-Autonomous-Vehicles \
  --repo-type dataset \
  --include "camera/camera_front_wide_120fov/*chunk_0001*" \
  --local-dir ./nvidia_dataset_raw

hf download nvidia/PhysicalAI-Autonomous-Vehicles \
  --repo-type dataset \
  --include "calibration/*chunk_0001*" \
  --local-dir ./nvidia_dataset_raw

hf download nvidia/PhysicalAI-Autonomous-Vehicles \
  --repo-type dataset \
  --include "labels/*chunk_0001*" \
  --local-dir ./nvidia_dataset_raw

hf download nvidia/PhysicalAI-Autonomous-Vehicles \
  --repo-type dataset \
  --include "lidar/*chunk_0001*" \
  --local-dir ./nvidia_dataset_raw

hf download nvidia/PhysicalAI-Autonomous-Vehicles \
  --repo-type dataset \
  --include "metadata/*" \
  --local-dir ./nvidia_dataset_raw

hf download nvidia/PhysicalAI-Autonomous-Vehicles \
  --repo-type dataset \
  --include "radar/radar_corner_front_left_srr_0/*chunk_0001*" \
  --local-dir ./nvidia_dataset_raw
```
