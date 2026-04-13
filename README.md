## Pre-requisites:

| Requirement | Details |
|---|---|
| **OS** | Linux |
| **GPU** | NVIDIA GPU with ≥ 6 GB VRAM |
| **NVIDIA Driver** | ≥ 470 |
| **CUDA Toolkit** | 11.1 |
| **Python** | 3.8 (for BEVFormer conda env) |
| **Conda** | Miniconda or Anaconda |
| **Git** | Installed |

## Downloading Dataset

Make an account in `https://www.nuscenes.org/download` and create an account.

## Expected Folder Structure

For nuScenes datasets:
```
/data/sets/nuscenes
    samples	-	Sensor data for keyframes.
    sweeps	-	Sensor data for intermediate frames.
    maps	-	Folder for all map files: rasterized .png images and vectorized .json files.
    v1.0-*	-	JSON tables that include all the meta data and annotations. Each split (trainval, test, mini) is provided in a separate folder.
```

## Installing dependencies
To properly work with the repo, run the following command from the root dir:
```
python -m venv .venv
source .venv/bin/activate       # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

## Download NUSCENES Data
The nuScenes data will be downloaded into the proper designated folder:
```
data/sets/nuscenes/
```
To download the nuScenes data, you need to run the `scripts/download_nuscenes.py` file.

To ensure this, create a `.env` file in the root with the following data:
```
NUSCENES_USERNAME="YOUR_NUSCENES_EMAIL"
NUSCENES_PASSWORD="YOUR_NUSCENES_PASSWORD"
```
Also, populate the `constants.py` file in the root with relevant data. More instructions can be found in the `constants.py` file

Run the following command from the root:
```
python scripts/download_nuscenes.py
```
[!IMPORTANT]
MD5 Integrity Troubleshooting > If the script reports an MD5 checksum mismatch, the file is likely corrupted or incomplete. To manually verify the file's hash and compare it against the value in constants.py, run:
```
md5sum data/sets/nuscenes/FILENAME
```

## Clone BEVFormer

From the root, run:
```bash
git clone https://github.com/fundamentalvision/BEVFormer.git core_models/BEVFormer
```

## Create BEVFormer Conda Environment

```bash
conda create -n bevformer python=3.8 -y
conda activate bevformer
```

## Install BEVFormer Dependencies (OpenMMLab Stack)

Run these commands **in order**, each step depends on the previous one:

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

pip install mmdet==2.14.0 mmsegmentation==0.14.1

# mmdetection3d from source (MUST be v0.17.1)
cd core_models
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
python setup.py install
cd ../../
```
> [!NOTE]
> If the host system lacks a compatible NVIDIA GPU or the necessary CUDA drivers, the installation or runtime may report `CUDA Error` or `Environment errors`. For the purposes of Perception Layer Analysis and OOD Monitoring logic, these errors can be treated as warnings, they prefer to run the perception layers on the CPU instead. In that case, the core Python logic and data processing modules will remain functional and can be executed on the CPU.

```bash
pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13  typing-extensions==4.5.0 pylint ipython==8.12  numpy==1.19.5 matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 scikit-image==0.19.3 setuptools==59.5.0
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
> [!NOTE]
> The `detectron2` installation may fail if the host system lacks a compatible NVIDIA GPU or the necessary CUDA drivers.

## Download Pre-trained Weights

All weights are stored in the project-level `checkpoints/` folder. BEVFormer accesses them via a symlink.

From the root, run:
```bash
# Download weights to the project checkpoints folder
wget -P checkpoints/ https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth

# Symlink BEVFormer's ckpts directory to project checkpoints
cd core_models/BEVFormer
ln -sv ../../checkpoints ckpts
cd ../../
```

## Prepare Data for BEVFormer (Symlink Strategy)

BEVFormer expects data at `BEVFormer/data/nuscenes/`, but our project stores it at the repo root (`data/sets/nuscenes/`). Instead of duplicating hundreds of GB of data, we use **symbolic links**.

### Create nuScenes Symlink

```bash
cd core_models/BEVFormer
mkdir -p data
ln -sv ../../../data/sets/nuscenes data/nuscenes
cd ../../
```


### Download CAN Bus Expansion Data

BEVFormer requires CAN bus sensor data from nuScenes:

1. Go to https://www.nuscenes.org/download
2. Download **`can_bus.zip`** (under "CAN bus expansion")
3. Extract it into BEVFormer's data directory:

```bash
unzip can_bus.zip -d core_models/BEVFormer/data/
```

### Generate Annotation PKL Files

BEVFormer uses **custom temporal annotation pickle files** (different from standard mmdet3d). Generate them before running inference:

```bash
cd core_models/BEVFormer
conda activate bevformer

# For full dataset (v1.0):
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version  v1.0-mini \ #change version according to dataset
    --canbus ./data
```

This generates:
```
data/nuscenes/
├── nuscenes_infos_temporal_train.pkl
└── nuscenes_infos_temporal_val.pkl
```

## Run BEVFormer (Inference / Evaluation)

### Single-GPU Evaluation

From the root, run:
```bash
cd core_models/BEVFormer
conda activate bevformer

# BEVFormer Tiny
python tools/test.py \
    projects/configs/bevformer/bevformer_tiny.py \
    ckpts/bevformer_tiny_epoch_24.pth \
    --eval bbox

# BEVFormer Base
python tools/test.py \
    projects/configs/bevformer/bevformer_base.py \
    ckpts/bevformer_r101_dcn_24ep.pth \
    --eval bbox
```

### Multi-GPU Evaluation (Cluster)

```bash
# BEVFormer Tiny with 8 GPUs
./tools/dist_test.sh \
    projects/configs/bevformer/bevformer_tiny.py \
    ckpts/bevformer_tiny_epoch_24.pth \
    8

# BEVFormer Base with 8 GPUs
./tools/dist_test.sh \
    projects/configs/bevformer/bevformer_base.py \
    ckpts/bevformer_r101_dcn_24ep.pth \
    8
```

> [!NOTE]
> Using **1 GPU** for evaluation gives slightly higher scores because continuous video sequences
> are not truncated across GPU boundaries.

