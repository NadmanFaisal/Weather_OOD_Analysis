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
The analysis scripts in this repo require **Python ≥ 3.11** (due to scipy, scikit-learn, etc.).

> [!IMPORTANT]
> This is a **separate environment** from the BEVFormer conda env (which uses Python 3.8). Make sure you use the correct environment for each task:
> - **`.venv`** (Python 3.11+) → for running analysis scripts, data processing, and `requirements.txt` packages
> - **`bevformer`** (Python 3.8) → for running BEVFormer inference only
From the root dir:
```
python -m venv .venv
source .venv/bin/activate       # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

<details>

  <summary>If your system Python is too old (e.g., Python 3.6–3.9)</summary>

  Use conda to create a Python 3.11 venv instead:

  ```bash
  conda create -n weather_ood python=3.11 -y
  conda activate weather_ood
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

  Or load a newer Python module if on an HPC cluster:
  ```bash
  module avail Python        # Find available versions
  module load Python/3.11.x  # Load a suitable version
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

</details>

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
> [!WARNING]
> Before proceeding with BEVFormer setup, **deactivate any existing virtual environment** to avoid conflicts:
> ```bash
> deactivate          # if .venv is active
> conda deactivate    # if another conda env is active
> ```
> Having `.venv` (Python 3.11+) and the BEVFormer conda env (Python 3.8) active at the same time will cause the wrong Python to be used, leading to package installation failures.
From the root, run:
```bash
git clone https://github.com/fundamentalvision/BEVFormer.git core_models/BEVFormer
```

## Create BEVFormer Conda Environment
> [!IMPORTANT]
> BEVFormer requires **Python 3.8** and **PyTorch 1.9.x**. PyTorch 1.9.x wheels only exist for Python 3.6–3.9. You **must** use conda with Python 3.8.

<details>

  <summary>If conda is not available on your system</summary>

  Install Miniconda locally (no root required). If your home directory has a storage or file count quota, install to a project/scratch directory with more space instead:

  ```bash
  # Download Miniconda
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

  # Install (change the -p path if your home dir has limited quota)
  bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

  # Optional: redirect conda cache to avoid filling home directory
  $HOME/miniconda3/bin/conda config --add pkgs_dirs $HOME/miniconda3/pkgs
  $HOME/miniconda3/bin/conda config --add envs_dirs $HOME/miniconda3/envs

  # Initialize and reload shell
  $HOME/miniconda3/bin/conda init bash
  source ~/.bashrc
  ```

</details>

```bash
conda create -n bevformer python=3.8 -y
conda activate bevformer
```
> [!WARNING]
> If `pip install` fails with `ERROR: Could not find an activated virtualenv (required)`, your system has a global pip config enforcing virtualenvs. Override it with a user config:
> ```bash
> mkdir -p ~/.config/pip
> echo -e "[install]\nrequire-virtualenv = false" > ~/.config/pip/pip.conf
> ```
## Install BEVFormer Dependencies (OpenMMLab Stack)

Run these commands **in order** from the root, each step depends on the previous one:

### Step 1: Verify Python version

```bash
python --version   # Must show Python 3.8.x
```

If this shows Python 3.10+, you are not in the conda environment. Run `conda activate bevformer` first.

### Step 2: Pin setuptools (prevents build errors with later steps)

```bash
pip install setuptools==59.5.0
```

### Step 3: Install PyTorch

> [!WARNING]
> The legacy `torch_stable.html` URL no longer serves old PyTorch wheels. You **must** use `--extra-index-url` as shown below.

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 --extra-index-url https://download.pytorch.org/whl/cu111
```

### Step 4: Install MMCV

```bash
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

### Step 5: Install MMDet + MMSeg

```bash
pip install mmdet==2.14.0 mmsegmentation==0.14.1
```
### Step 6: Set up CUDA for compilation

mmdetection3d requires compiling CUDA extensions. You need `CUDA_HOME` set and a compatible GCC version.

**Option A — Use a system CUDA module** (recommended for HPC clusters):
```bash
module load CUDA/11.3.1          # or any CUDA 11.x available on your system
export CUDA_HOME=$CUDA_ROOT      # $CUDA_ROOT is set by the module
```

**Option B — Use conda CUDA** (if no system CUDA is available):
```bash
# IMPORTANT: install CUDA 11.x specifically, NOT the latest version
conda install -c nvidia cuda-toolkit=11.8 -y
export CUDA_HOME=$CONDA_PREFIX
```
> [!WARNING]
> **Do NOT install the latest `cuda-toolkit` via conda** (i.e., without pinning a version). Recent versions ship with CCCL/Thrust headers that require C++17, which is incompatible with mmdetection3d v0.17.1 (compiled with C++14). Always pin to **CUDA 11.x**.

Verify CUDA is accessible:
```bash
echo $CUDA_HOME    # Should print a path
nvcc --version     # Should show CUDA 11.x
```
### Step 7: Handle GCC compatibility

CUDA 11.x requires **GCC ≤ 10**. Check your version:

```bash
gcc --version
```

If GCC is **version 11 or higher** (which is common on modern systems), install a compatible version via conda:

```bash
conda install -c conda-forge "gcc_linux-64<11" "gxx_linux-64<11" -y
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
```

If your GCC is already version 10 or lower, you can skip this step.

### Step 8: Pre-install pinned dependencies

These must be installed **before** building mmdetection3d to avoid version conflicts:

```bash
pip install trimesh==2.35.39 tensorboard==2.11.0 scikit-image==0.19.3
```
### Step 9: Build and install mmdetection3d
```bash
cd core_models
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1
pip install -e . --no-deps
cd ../../
```
> [!NOTE]
> We use `pip install -e . --no-deps` for two reasons:
> - **`-e`** (editable mode) avoids legacy `easy_install` dependency resolution issues that cause `setuptools`/`grpcio` build failures.
> - **`--no-deps`** skips automatic dependency resolution, preventing version conflicts (e.g., mmdetection3d's `plyfile` dependency requires numpy≥1.21, but BEVFormer needs numpy==1.19.5). All required dependencies are manually installed in Steps 8 and 10.

<details>

  <summary>If compilation fails with missing crypt.h</summary>

  ```bash
  conda install -c conda-forge libxcrypt -y
  pip install -e . --no-deps
  ```

</details>

<details>

  <summary>If compilation fails with cstdint error</summary>

  ```bash
  find $CONDA_PREFIX/lib/python3.8/site-packages/torch/include \
    -name "*.h" -exec grep -l "uint16_t\|uint32_t" {} \; | \
    xargs -I{} sed -i '1i #include <cstdint>' {}

  pip install -e . --no-deps
  ```

</details>

### Step 10: Install remaining BEVFormer dependencies
From root, run the following:
```bash
pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13 pylint ipython==8.12 numba==0.48.0 pandas==1.4.4 pyquaternion shapely fire cachetools scikit-learn
```
### Step 11: Install detectron2

Use the **prebuilt wheel** (recommended — building from source often fails with torch 1.9.0):

```bash
pip install https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/detectron2-0.6%2Bcu111-cp38-cp38-linux_x86_64.whl
```

<details>

  <summary>If the prebuilt wheel fails or you use a different Python version</summary>

  Browse the [wheel index](https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html) and pick the correct `.whl` for your Python version (cp37, cp38, or cp39).

  Or install from source with a pinned version compatible with torch 1.9.0:

  ```bash
  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
  ```

  > [!WARNING]
  > Do **not** install detectron2 from `main` branch (`git+https://github.com/facebookresearch/detectron2.git`). The latest code requires newer PyTorch and will fail to compile with torch 1.9.0.

</details>

### Step 12: Re-pin dependency versions

detectron2 may overwrite or remove some dependency versions during installation. Re-pin and re-install them:

```bash
pip install numpy==1.19.5 matplotlib==3.5.2 typing-extensions==4.5.0 Pillow==9.5.0 setuptools==59.5.0 pyquaternion shapely fire cachetools scikit-learn
```

Verify the environment is working:
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import mmcv; print('MMCV:', mmcv.__version__)"
python -c "import mmdet; print('MMDet:', mmdet.__version__)"
python -c "import mmdet3d; print('MMDet3D:', mmdet3d.__version__)"
python -c "import detectron2; print('Detectron2:', detectron2.__version__)"
```
## Download Pre-trained Weights

All weights are stored in the project-level `checkpoints/` folder. BEVFormer accesses them via a symlink.

From the root, run:
```bash
# Run the download script from the repo root (Weather_OOD_Analysis):
python scripts/download_weights.py

# You should see an output like this: 
=== BEVFormer Weights ===
  [OK]  BEVFormer main checkpoint (r101, 24ep)
  [OK]  BEVFormer tiny checkpoint (r50, 24ep)
  [OK]  BEVFormer backbone pretrain (ResNet-101 DCN)

=== BEVFusion Weights ===
  [OK]  BEVFusion det checkpoint (official mit-han-lab Dropbox)
  [OK]  BEVFusion Swin-T backbone pretrain

All weights downloaded successfully. ✓
```
```bash
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
BEVFormer requires CAN bus sensor data from nuScenes. This is downloaded automatically using your `.env` credentials (same ones used for `download_nuscenes.py`):
```bash
python scripts/download_canbus.py
```
This downloads `can_bus.zip`, extracts it to `core_models/BEVFormer/data/can_bus/`, and cleans up the zip file.
### Generate Annotation PKL Files

BEVFormer uses **custom temporal annotation pickle files** (different from standard mmdet3d). Generate them before running inference:

```bash
conda activate bevformer
cd core_models/BEVFormer

# Make tools a proper Python package (required for imports)
touch tools/__init__.py
touch tools/data_converter/__init__.py

# For mini dataset (v1.0):
PYTHONPATH=. python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version  v1.0-mini \
    --canbus ./data
```
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

> [!IMPORTANT]
> BEVFormer **requires a GPU** and uses PyTorch distributed mode for all evaluations — even single-GPU runs. You must use `dist_test.sh`, not `python tools/test.py` directly.

### Step 1: Request a GPU compute node

Do **not** run inference on the login node — it will fail with `AssertionError`.

**Interactive session** (for testing and debugging):
```bash
# Replace NAISS202X-X-X with your project allocation (run `projinfo` to find it)
# Replace T4:1 with the GPU type and count available on your cluster
srun --account=NAISS2026-X-X --gpus-per-node=T4:1 --time=01:00:00 --pty /bin/bash
```

<details>

  <summary>How to find your project allocation</summary>

  ```bash
  projinfo     # Shows your project ID, usage, and available hours
  ```

</details>

### Step 2: Set up the environment on the GPU node
Once on the GPU node, set up the environment:
```bash
conda activate bevformer
module load CUDA/11.3.1          # or your CUDA 11.x module
export CUDA_HOME=$CUDA_ROOT
cd /path/to/Weather_OOD_Analysis/core_models/BEVFormer
```
Verify GPU access:
```bash
nvidia-smi    # Should show your allocated GPU(s)
```
### Step 3: Configure the annotation file
Depending on which dataset you downloaded, edit the BEVFormer config file to point to the correct `.pkl` files:

- Open `projects/configs/bevformer/bevformer_base.py` (or `bevformer_tiny.py`)
- Find `ann_file` entries (~line 197) and update them to match your generated `.pkl` files:
  - Mini dataset: `nuscenes_infos_temporal_train.pkl` → `nuscenes_infos_mini_train.pkl`
  - Full dataset: keep as-is (`nuscenes_infos_temporal_train.pkl`)
### Step 4: Run evaluation
**Single-GPU evaluation:**
```bash
PYTHONPATH=. ./tools/dist_test.sh \
    projects/configs/bevformer/bevformer_base.py \
    ckpts/bevformer_r101_dcn_24ep.pth \
    1 \
    --eval bbox
```
**Multi-GPU evaluation** (request multiple GPUs in your `srun` command first):
```bash
# BEVFormer Base with 4 GPUs
PYTHONPATH=. ./tools/dist_test.sh \
    projects/configs/bevformer/bevformer_base.py \
    ckpts/bevformer_r101_dcn_24ep.pth \
    4 \
    --eval bbox
```
**BEVFormer Tiny** (lighter model, faster inference):
```bash
PYTHONPATH=. ./tools/dist_test.sh \
    projects/configs/bevformer/bevformer_tiny.py \
    ckpts/bevformer_tiny_epoch_24.pth \
    1 \
    --eval bbox
```
> [!NOTE]
> - The last number (1, 4, 8) must match the number of GPUs you requested in `srun`.
> - Using **1 GPU** gives slightly higher scores because continuous video sequences are not truncated across GPU boundaries.
> - Always use `PYTHONPATH=.` to ensure BEVFormer's custom modules are importable.
> - When done, type `exit` to release the GPU node and stop billing your allocation.
## Preparing Mixed Corruption and Clean data
To create the corrupted data with the current simple corruption script, run the following command:
```bash
python scripts/generate_corrupted_data.py
```
> [!NOTE]
> To create different types of corruption, look into `constants.py` in the root and the `CORRUPTION_MAP` variable,
This will generate corruption data as follows:
```
data/sets/nuscenes_corrupted/
├── fog/
└── snow/
```
But only creating these will not suffice, you also need to make them have the perfect folder structure as required by nuScenes. Perform the following command from the `root`:
```bash
cd data/sets/nuscenes_corrupted/snow

# Link the core metadata and temporal sweeps
ln -sfn ../../nuscenes/v1.0-mini v1.0-mini
ln -sfn ../../nuscenes/maps maps

# Enter the sweeps folder
cd sweeps

ln -sfn ../../../nuscenes/sweeps/LIDAR_TOP LIDAR_TOP
ln -sfn ../../../nuscenes/sweeps/RADAR_FRONT RADAR_FRONT
ln -sfn ../../../nuscenes/sweeps/RADAR_FRONT_LEFT RADAR_FRONT_LEFT
ln -sfn ../../../nuscenes/sweeps/RADAR_FRONT_RIGHT RADAR_FRONT_RIGHT
ln -sfn ../../../nuscenes/sweeps/RADAR_BACK_LEFT RADAR_BACK_LEFT
ln -sfn ../../../nuscenes/sweeps/RADAR_BACK_RIGHT RADAR_BACK_RIGHT

cd ../


# Enter the samples folder (where your snowy CAM images are)
cd samples

# Link the LiDAR and Radar sensors from the clean dataset
ln -sfn ../../../nuscenes/samples/LIDAR_TOP LIDAR_TOP
ln -sfn ../../../nuscenes/samples/RADAR_FRONT RADAR_FRONT
ln -sfn ../../../nuscenes/samples/RADAR_FRONT_LEFT RADAR_FRONT_LEFT
ln -sfn ../../../nuscenes/samples/RADAR_FRONT_RIGHT RADAR_FRONT_RIGHT
ln -sfn ../../../nuscenes/samples/RADAR_BACK_LEFT RADAR_BACK_LEFT
ln -sfn ../../../nuscenes/samples/RADAR_BACK_RIGHT RADAR_BACK_RIGHT

# Return to root
cd ../../../../../
```
> [!Note]
> The above commands show an example for only one type of corruption folder (`snow`). There might be multiple corruptions folder (`fog, rain, etc.`). Make sure to change the commands and create the ghost folders accordingly.

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
        ann_file=data_root + 'nuscenes/nuscenes_infos_temporal_train.pkl',
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

> [!Note] on nuScenes Evaluation Crashes with Mixed Data
> If you run the standard BEVFormer evaluation script (tools/test.py --eval bbox) on a custom-mixed dataset, the model will successfully process the images, but the script will inevitably crash at the very end with the following error:
> AssertionError: Samples in split doesn't match samples in predictions.

> What is happening here?
> This crash is caused by a strict, hardcoded rule within the official nuscenes-devkit evaluation code. When the nuScenes Grader calculates the official mAP (mean Average Precision) scores, it requires a perfect 1-to-1 match between the number of frames predicted by the model and the number of ground-truth frames in the official validation database (which contains exactly 81 frames for v1.0-mini).

> Because we generated a custom dataset consisting of 50 clean and 50 corrupted frames, the lengths do not match. Additionally, because BEVFormer requires a continuous sequence of temporal frames to build its memory, our randomized shuffling breaks the expected timeline. BEVFormer correctly drops the frames where it lacks sufficient temporal history (resulting in roughly 67 usable frames), causing the final size mismatch that crashes the evaluator.

> Why we do not care about this final evaluation:
For the scope of our Out-of-Distribution (OOD) analysis, we do not care about official nuScenes object detection benchmarking. The official evaluator only calculates how accurately the model drew 3D bounding boxes. However, to calculate OOD metrics like Energy Scores and Mahalanobis Distance, we need the model's raw mathematical uncertainty—specifically, the logits and feature maps generated inside the classification head before the bounding boxes are finalized.

> The fact that BEVFormer successfully processed the 67 mixed frames means the model did its job perfectly. To extract our required metrics, we completely bypass the rigid nuScenes grading script. Instead, we use a custom Python script equipped with PyTorch Forward Hooks to quietly intercept and save the raw logits as the model runs, allowing us to evaluate its performance against weather corruption mathematically.
