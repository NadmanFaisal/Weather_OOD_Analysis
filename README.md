## Pre-requisites:

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

BEVFormer/data symlink create:
```
cd core_models/BEVFormer
mkdir -p data
ln -sv ../../../data/sets/nuscenes data/nuscenes
```

Installing conda for BEVFormer
```
conda create -n bevformer python=3.8 -y
conda activate bevformer

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0 mmsegmentation==0.14.1
```
Go back to `core_models/` directory
```
cd core_models
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
python setup.py install
```
