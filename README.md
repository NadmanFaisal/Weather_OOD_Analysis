## Pre-requisites:

- [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli#download-a-dataset-or-a-space)
- [HuggingFace access token](https://huggingface.co/settings/tokens)

## How to Download Data from HuggingFace

Make sure to go through the data file structure in hugging face to understand what files to download. Also ensure that the huggingface cli is installed and authenticated in your system. Use the following commands to do so:

```
hf --help
hf auth login
```
The following command will download only chunk_0001 of the NVIDIA autonomous vehicle dataset. 

---
**NOTE**

The data will be stored in `nvidia_dataset_raw` directory within your working directory. Change the `chunk_0001` to some other value to download some other chunk: 

---
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

```
pip install physical_ai_av
```
