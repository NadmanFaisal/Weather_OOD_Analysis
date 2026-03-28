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
