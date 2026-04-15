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
> [!IMPORTANT]
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
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 --extra-index-url https://download.pytorch.org/whl/cu111

pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

pip install mmdet==2.14.0 mmsegmentation==0.14.1

# mmdetection3d from source (MUST be v0.17.1)
cd core_models
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
python setup.py install
```

<details>

  <summary>If you face errors</summary>

  ```bash
  conda install -c nvidia cuda-toolkit
  export CUDA_HOME=$CONDA_PREFIX

  find $CONDA_PREFIX/lib/python3.8/site-packages/torch/include \
    -name "*.h" -exec grep -l "uint16_t\|uint32_t" {} \; | \
    xargs -I{} sed -i '1i #include <cstdint>' {}

  pip install trimesh==2.35.39 tensorboard==2.11.0 scikit-image==0.19.3
  TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" python setup.py develop
  ```

</details>

From root, run the following:
```bash
pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13  typing-extensions==4.5.0 pylint ipython==8.12  numpy==1.19.5 matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 scikit-image==0.19.3 setuptools==59.5.0
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

<details>

  <summary>If you face errors</summary>
  
  Use the prebuilt wheel instead:

  ```bash
  pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
  ```

  Also fix Pillow and missing deps:

  ```bash
  pip install Pillow==9.5.0
  pip install pyquaternion shapely fire cachetools scikit-learn
  ```

</details>

## Download Pre-trained Weights

All weights are stored in the project-level `checkpoints/` folder. BEVFormer accesses them via a symlink.

From the root, run:
```bash
# Run the download script from the repo root (Weather_OOD_Analysis):
python scripts/download_weights.py

# You should see an output like this: 
=== BEVFormer Weights ===
  [OK]  BEVFormer main checkpoint (r101, 24ep)
  [OK]  BEVFormer backbone pretrain (ResNet-101 DCN)

=== BEVFusion Weights ===
  [OK]  BEVFusion det checkpoint (official mit-han-lab Dropbox)
  [OK]  BEVFusion Swin-T backbone pretrain

All weights downloaded successfully. ✓

# Symlink BEVFormer's ckpts directory to project checkpoints
cd core_models/BEVFormer
ln -sv ../../checkpoints/bevformer ckpts
cd ../../
```

## Prepare Data for BEVFormer (Symlink Strategy)

BEVFormer expects data at `BEVFormer/data/nuscenes/`, but our project stores it at the repo root (`data/sets/nuscenes/`). Instead of duplicating hundreds of GB of data, we use **symbolic links**.

### Create nuScenes Symlink

```bash
cd core_models/BEVFormer
mkdir -p data
cd data
ln -sv ../../../data/sets/{nuscenes,nuscenes_corrupted,nuscenes_combined} .
cd ../../../
```
Do the following to link proper folders necessary during evaluation phase:
```bash
cd core_models/BEVFormer/data
ln -sfn nuscenes/v1.0-mini v1.0-mini
ln -sfn nuscenes/maps maps
cd ../../../
```
### Download CAN Bus Expansion Data

BEVFormer requires CAN bus sensor data from nuScenes:

1. Go to https://www.nuscenes.org/download
2. Download **`can_bus.zip`** (under "CAN bus expansion")
3. Place the zip file in the root
3. Extract it into BEVFormer's data directory:
4. (Optional) You can delete the `can_bus.zip` from the root.

```bash
unzip can_bus.zip -d core_models/BEVFormer/data/
```

### Generate Annotation PKL Files

BEVFormer uses **custom temporal annotation pickle files** (different from standard mmdet3d). Generate them before running inference:

```bash
conda activate bevformer
cd core_models/BEVFormer

# For mini dataset (v1.0):
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version  v1.0-mini \
    --canbus ./data
```

<details>

  <summary>If you face errors</summary>
  
  Convert the tool folder into a python package:

  ```bash
  touch tools/__init__.py

  PYTHONPATH=. python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0-mini \
    --canbus ./data
  ```

</details>

> [!NOTE]
Change the `--version` flag to `'v1.0-trainval'` if using the full dataset. Depending on the version used, this generates index files in your `core_models/BEVFormer/data/nuscenes/` directory, such as:
```
data/nuscenes/
├── nuscenes_infos_temporal_train.pkl
└── nuscenes_infos_temporal_val.pkl
```
Or:
```
data/nuscenes/
├── nuscenes_infos_mini_train.pkl
└── nuscenes_infos_mini_val.pkl
```
## Run BEVFormer (Inference / Evaluation)

### Single-GPU Evaluation

From the root, run:
```bash
cd core_models/BEVFormer
conda activate bevformer
```
Depending on which dataset you use, you need to go to `core_models/BEVFormer/projects/configs/bevformer/[whichever_model_you_want_to_train]`.
Navigate to the dictionary at line 197 and change the fields for `ann_file=data_root + 'nuscenes_infos_temporal_train.pkl'` to the respective `.pkl` files generated in the above steps.

To run BEVFormer Tiny model:
```bash
python tools/test.py \
    projects/configs/bevformer/bevformer_tiny.py \
    ckpts/bevformer_tiny_epoch_24.pth \
    --eval bbox
```

<details>

  <summary>If you face errors</summary>
  
  Fix the path for the test folder:

  ```bash
  PYTHONPATH=. python tools/test.py \
    projects/configs/bevformer/bevformer_tiny.py \
    ckpts/bevformer_tiny_epoch_24.pth \
    --eval bbox
  ```

</details>

To run BEVFormer Base version:
```bash
python tools/test.py \
    projects/configs/bevformer/bevformer_base.py \
    ckpts/bevformer_r101_dcn_24ep.pth \
    --eval bbox
```

### Multi-GPU Evaluation (Cluster)
To run BEVFormer Tiny with 8 GPUs
```bash
./tools/dist_test.sh \
    projects/configs/bevformer/bevformer_tiny.py \
    ckpts/bevformer_tiny_epoch_24.pth \
    8
```
To run BEVFormer Base with 8 GPUs
```bash
# BEVFormer Base with 8 GPUs
./tools/dist_test.sh \
    projects/configs/bevformer/bevformer_base.py \
    ckpts/bevformer_r101_dcn_24ep.pth \
    8
```

> [!NOTE]
> Using **1 GPU** for evaluation gives slightly higher scores because continuous video sequences
> are not truncated across GPU boundaries.

## Preparing Mixed Corruption and Clean data
To create the corrupted data with the current simple corruption script, run the following command:
```bash
python scripts/generate_corrupted_data.py
```
> ![NOTE]
> To create different types of corruption, look into `constants.py` in the root and the `CORRUPTION_MAP` variable,
This will generate corruption data as follows:
```
data/sets/nuscenes_corrupted/
├── fog/
└── snow/
```
Now, you will need to create the pkl files for the new corrupted data. Do this by doing the following command:
```bash
conda activate bevformer
cd core_models/BEVFormer

# For mini dataset (v1.0):
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes_corrupted/fog \
    --out-dir ./data/nuscenes_corrupted/fog \
    --extra-tag nuscenes \
    --version  v1.0-mini \
    --canbus ./data
```

<details>

  <summary>If you face errors</summary>
  
  Convert the tool folder into a python package:

  ```bash
  touch tools/__init__.py

  PYTHONPATH=. python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes_corrupted/fog \
    --out-dir ./data/nuscenes_corrupted/fog \
    --extra-tag nuscenes \
    --version v1.0-mini \
    --canbus ./data
  ```

</details>

> [!NOTE]
Change the `--version` flag to `'v1.0-trainval'` if using the full dataset. Depending on the version used, this generates index files in your `core_models/BEVFormer/data/nuscenes_corrupted/fog/` directory, such as:
```
data/nuscenes/
├── nuscenes_infos_temporal_train.pkl
└── nuscenes_infos_temporal_val.pkl
```
Or:
```
data/nuscenes/
├── nuscenes_infos_mini_train.pkl
└── nuscenes_infos_mini_val.pkl
```
After these, run the following command to create a mix of 50 corrupted data and 50 clean data:
```bash
python scripts/create_mixed_data.py
```
This will generate corruption data as follows:
```
data/sets/nuscenes_combined/
└── ..._MIXED_.pkl
```

Now to feed the mixed dataset to the `bevformer_tiny` model, you will have to navigate to `core_models/BEVFormer/project/bevformer/bevformer_tiny`, and paste the following from line 172 - 239:
```python
dataset_type = 'CustomNuScenesDataset'
data_root = 'data/'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
   
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'nuscenes_combined/nuscenes_infos_temporal_MIXED_val.pkl',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_combined/nuscenes_infos_temporal_MIXED_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)
```
That being done, run the following command:
To run BEVFormer Tiny model:
```bash
python tools/test.py \
    projects/configs/bevformer/bevformer_tiny.py \
    ckpts/bevformer_tiny_epoch_24.pth \
    --eval bbox
```

<details>

  <summary>If you face errors</summary>
  
  Fix the path for the test folder:

  ```bash
  PYTHONPATH=. python tools/test.py \
    projects/configs/bevformer/bevformer_tiny.py \
    ckpts/bevformer_tiny_epoch_24.pth \
    --eval bbox
  ```

</details>

To run BEVFormer Base version:
```bash
python tools/test.py \
    projects/configs/bevformer/bevformer_base.py \
    ckpts/bevformer_r101_dcn_24ep.pth \
    --eval bbox
```

> ![Note] on nuScenes Evaluation Crashes with Mixed Data
> If you run the standard BEVFormer evaluation script (tools/test.py --eval bbox) on a custom-mixed dataset, the model will successfully process the images, but the script will inevitably crash at the very end with the following error:
> AssertionError: Samples in split doesn't match samples in predictions.

> What is happening here?
> This crash is caused by a strict, hardcoded rule within the official nuscenes-devkit evaluation code. When the nuScenes Grader calculates the official mAP (mean Average Precision) scores, it requires a perfect 1-to-1 match between the number of frames predicted by the model and the number of ground-truth frames in the official validation database (which contains exactly 81 frames for v1.0-mini).

> Because we generated a custom dataset consisting of 50 clean and 50 corrupted frames, the lengths do not match. Additionally, because BEVFormer requires a continuous sequence of temporal frames to build its memory, our randomized shuffling breaks the expected timeline. BEVFormer correctly drops the frames where it lacks sufficient temporal history (resulting in roughly 67 usable frames), causing the final size mismatch that crashes the evaluator.

> Why we do not care about this final evaluation:
For the scope of our Out-of-Distribution (OOD) analysis, we do not care about official nuScenes object detection benchmarking. The official evaluator only calculates how accurately the model drew 3D bounding boxes. However, to calculate OOD metrics like Energy Scores and Mahalanobis Distance, we need the model's raw mathematical uncertainty—specifically, the logits and feature maps generated inside the classification head before the bounding boxes are finalized.

> The fact that BEVFormer successfully processed the 67 mixed frames means the model did its job perfectly. To extract our required metrics, we completely bypass the rigid nuScenes grading script. Instead, we use a custom Python script equipped with PyTorch Forward Hooks to quietly intercept and save the raw logits as the model runs, allowing us to evaluate its performance against weather corruption mathematically.
