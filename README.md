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
