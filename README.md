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
pip install lyft_dataset_sdk nuscenes-devkit plyfile networkx==2.2
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
ln -sfn ../../../data/sets/nuscenes nuscenes
ln -sfn ../../../data/sets/nuscenes-c/nuScenes-c nuScenes-c
cd ../../../
```
Do the following to link proper folders necessary during evaluation phase:
```bash
cd core_models/BEVFormer/data
#ln -sfn nuscenes/v1.0-mini v1.0-mini
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

# For full dataset (v1.0):
PYTHONPATH=. python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version  v1.0 \
    --canbus ./data
```
> [!NOTE]
Change the `--version` flag to `'v1.0-mini'` if using the mini dataset. Depending on the version used, this generates index files in your `core_models/BEVFormer/data/nuscenes/` directory, such as:
```
data/nuscenes/
├── nuscenes_infos_temporal_train.pkl
└── nuscenes_infos_temporal_val.pkl
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
### Step 3: Run evaluation
**Single-GPU evaluation:**
```bash
OOD_WEATHER=Clear OOD_SEVERITY=baseline PYTHONPATH=. ./tools/dist_test.sh \
    projects/configs/bevformer/bevformer_base.py \
    ckpts/bevformer_r101_dcn_24ep.pth \
    1 \
    --eval bbox
```
### Step 4: Check evaluation results
You can check the evaluation results at the `test/bevformer_base/[DATE]/pts_bbox` directory.

**Multi-GPU evaluation** (request multiple GPUs in your `srun` command first):
```bash
# BEVFormer Base with 4 GPUs
OOD_WEATHER=Clear OOD_SEVERITY=baseline PYTHONPATH=. ./tools/dist_test.sh \
    projects/configs/bevformer/bevformer_base.py \
    ckpts/bevformer_r101_dcn_24ep.pth \
    4 \
    --eval bbox
```
### Step 4: Check evaluation results
You can check the evaluation results at the `test/bevformer_base/[DATE]/pts_bbox` directory.
> [!IMPORTANT]
> - The `OOD_WEATHER` and `OOD_SEVERITY` tells BEVFormer which folder to target (Case sensitive).
> - The `OOD_WEATHER` can be `Clear` for clean dataset, or `Fog` or `Snow` (for corrupted dataset).
> - The `OOD_SEVERITY` can be `baseline` for clean dataset, or `easy`, `mid`, or `hard` (for corrupted dataset).

> [!NOTE]
> - The last number (1, 4, 8) must match the number of GPUs you requested in `srun`.
> - Using **1 GPU** gives slightly higher scores because continuous video sequences are not truncated across GPU boundaries.
> - Always use `PYTHONPATH=.` to ensure BEVFormer's custom modules are importable.
> - When done, type `exit` to release the GPU node and stop billing your allocation.

## Downloading Corrupted data from OpenDataLab
This section will go through the specifics for downloading `nuScenes-c` data.
### Step 1: Installing dependencies
To download the corrupted `nuScenes-c` data created from Robo3D, you first need to do the following:
```bash
pip install openxlab
pip install python-dotenv
```
### Step 2:
Create an acc: https://openxlab.org.cn/home, and store the secrets in the `.env` in root as follows:
```bash
OPENXLAB_AK="YOUR_SECRET_ACTION_KEY"
OPENXLAB_SK="YOUR_SECRET_KEY"
```
### Step 3: Downloading the data
From the root run:
```bash
python scripts/download_nuscenes_c.py
```
The above instruction will create a folder `nuscenes-c` inside `data` directory as follows:
```
data/
└── sets/
    ├── OpenDataLab___nuScenes-C
    ├── nuscenes
    └── nuscenes-c/
        ├── nuScenes-C
        └── nuScenes-c/
            ├── Brightness
            ├── CameraCrash
            ├── ColorQuant
            ├── Fog/
            │   ├── easy
            │   ├── hard
            │   └── mid
            ├── FrameLost
            ├── LowLight
            ├── MotionBlur
            └── Snow
```
> [!NOTE]
> The corrupted datasets downloaded from OpenDataLab do not naturally come inside a `samples/` folder. We ensure in the later steps that the raw `CAM_FRONT`, `CAM_BACK`, etc., folders are physically placed inside `[Weather]/[Severity]/samples/` so that the perception models can find them.
### Step 4: Creating necessary directories
Run the following to create shadow folder:
```bash
python scripts/build_shadow_nuscenes.py 
```
> [!IMPORTANT]
> BEVFormer models expect a strict, unified folder structure (including `maps`, `sweeps`, and metadata JSONs). However, the nuScenes-c dataset downloaded from Robo3D is intentionally missing these folders. Because the benchmark's goal is to test Out-Of-Distribution (OOD) generalization without retraining, the dataset authors only provided the corrupted validation camera images, omitting the massive training sets and structural files to save space.
>
> To satisfy the PyTorch dataloader without breaking OOD rules, this script creates a "shadow" directory. It creates symlinks to the clean nuscenes dataset for structural requirements (like `maps` and `v1.0-trainval`), while reserving the `samples/` folder exclusively for the physical, corrupted .jpg images.

## Run BEVFormer with corrupted data (Inference / Evaluation)
Before we can forward feed the data into the perception models, we need to change a few things.
### Step 1: Understanding the directories
In our `core_models` directory, we have the BEVFormer directory:
```
core_models/
└── BEVFormer/
```
The `BEVFormer` folder is used to run models with any dataset (`Fog` corruptions, `Snow` corruptions, and also `nusecnes` clean dataset). We have made changes to the `config` of only the base model (`core_models/BEVFormer/projects/configs/bevformer/bevformer_base.py`), and also `tools` where the `PyTorch hooks` are attached for necessary interceptions.
### Step 2: Creating required PKL files
Now that we have understood the directories, we now need to make the necessary `pkl`. From the root::
```bash
python scripts/patch_pkl.py
```
> [!Important]
>
> What does it do?
>
> Since BEVFormer cannot manually annotate the corrupted data as training images are missing (only val set images are present), we need to create copies of the clean data `nuscenes`'s `pkl` files and then manually change the paths specified inside to point towards the corrupted images in the correct directories.
### Step 3: Request a GPU compute node

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

### Step 4: Set up the environment on the GPU node
Once on the GPU node, set up the environment:
```bash
conda activate bevformer
module load CUDA/11.3.1          # or your CUDA 11.x module
export CUDA_HOME=$CUDA_ROOT
```
Verify GPU access:
```bash
nvidia-smi    # Should show your allocated GPU(s)
```
### Step 5: Run evaluation
**Single-GPU evaluation:**
```bash
OOD_WEATHER=Snow OOD_SEVERITY=hard PYTHONPATH=. ./tools/dist_test.sh \
    projects/configs/bevformer/bevformer_base.py \
    ckpts/bevformer_r101_dcn_24ep.pth \
    1 \
    --eval bbox
```
**Multi-GPU evaluation** (request multiple GPUs in your `srun` command first):
```bash
# BEVFormer Base with 4 GPUs
OOD_WEATHER=Snow OOD_SEVERITY=hard PYTHONPATH=. ./tools/dist_test.sh \
    projects/configs/bevformer/bevformer_base.py \
    ckpts/bevformer_r101_dcn_24ep.pth \
    4 \
    --eval bbox
```
> [!IMPORTANT]
> - The `OOD_WEATHER` and `OOD_SEVERITY` tells BEVFormer which folder to target (Case sensitive).
> - The `OOD_WEATHER` can be `Fog` or `Snow` (for corrupted dataset), or `Clear` for clean dataset.
> - The `OOD_SEVERITY` can be `easy`, `mid`, or `hard` (for corrupted dataset) or `baseline` for clean dataset.

> [!NOTE]
> - The last number (1, 4, 8) must match the number of GPUs you requested in `srun`.
> - Using **1 GPU** gives slightly higher scores because continuous video sequences are not truncated across GPU boundaries.
> - Always use `PYTHONPATH=.` to ensure BEVFormer's custom modules are importable.
> - When done, type `exit` to release the GPU node and stop billing your allocation.
### Step 6: Check evaluation results
You can check the evaluation results at the `test/bevformer_base/[DATE]/pts_bbox` directory.
## Mahalanobis Distance and Energy Scores
When the evaluation phases are done, logits and latent feature maps are intercepted and stored under `data/intercepted_feature_logits` directory under our root.
### Generate Energy Scores
To generate energy scores, run the following command:
```bash
OOD_WEATHER=Weather OOD_SEVERITY=severity OOD_TIMESTAMP=DATE_TIME python safety_monitor/energy_score.py
```
This will generate energy scores (`.json`) and store them under `data/energy_scores` under our root.
> [!IMPORTANT]
> - The `OOD_WEATHER` and `OOD_SEVERITY` tells BEVFormer which folder to target (Case sensitive).
> - The `OOD_WEATHER` can be `Fog` or `Snow` (for corrupted dataset), or `Clear` for clean dataset.
> - The `OOD_SEVERITY` can be `easy`, `mid`, or `hard` (for corrupted dataset) or `baseline` for clean dataset.
> - For `OOD_TIMESTAMP`, please check what timestap you will use from `data/intercepted_feature_logits/{target_folder}/{timestamp}`
### Generate Mahalanobis Distances
To generate **Raw** baseline Mahalanobis distance:
```bash
OOD_WEATHER=Clear OOD_SEVERITY=baseline OOD_TIMESTAMP=DATE_TIME python safety_monitor/mahalanobis.py
```
This will generate the baseline mathematical parameters (`mahalanobis_baseline.pt`) and store them under `data/features/nuscenes/{timestamp}`. It will also generate the baseline score evaluations and store them under `data/mahalanobis_scores/nuscenes/{timestamp}`.

To generate **Normalized** Mahalanobis distance:
```bash
NORMALIZATION=true OOD_WEATHER=Clear OOD_SEVERITY=baseline OOD_TIMESTAMP=DATE_TIME python safety_monitor/mahalanobis.py
```
This will generate the baseline mathematical parameters (`mahalanobis_baseline.pt`) and store them under `data/features/nuscenes/{timestamp}`. It will also generate the baseline score evaluations and store them under `data/mahalanobis_scores/nuscenes/{timestamp}`.
> [!IMPORTANT]
> - The `NORMALIZATION` variable defaults to false. Setting it to true applies Global Average Pooling and L2 Normalization to the feature vectors, and automatically routes all saved files to a dedicated `normalized/` sub-folder to prevent overwriting your raw data.
> - The `OOD_WEATHER` and `OOD_SEVERITY` tells BEVFormer which folder to target (Case sensitive).
> - The `OOD_WEATHER` can be `Fog` or `Snow` (for corrupted dataset), or `Clear` for clean dataset.
> - The `OOD_SEVERITY` can be `easy`, `mid`, or `hard` (for corrupted dataset) or `baseline` for clean dataset.
> - For `OOD_TIMESTAMP`, please check what timestap you will use from `data/intercepted_feature_logits/nuscenes/{timestamp}`
To generate Mahalanobis distances of corrupted data (or also clean dataset), Raw or Normalized:
```bash
# Raw
OOD_WEATHER=[WEATHER] OOD_SEVERITY=[SEVERITY] OOD_TIMESTAMP=[DATE_TIME] BASELINE_TIMESTAMP=[DATE_TIME] python safety_monitor/mahalanobis.py

#Normalized
NORMALIZATION=true OOD_WEATHER=[WEATHER] OOD_SEVERITY=[SEVERITY] OOD_TIMESTAMP=[DATE_TIME] BASELINE_TIMESTAMP=[DATE_TIME] python safety_monitor/mahalanobis.py
```
This will output the final raw evaluated `.json` scores to `data/mahalanobis_scores/{OOD_WEATHER}/{OOD_SEVERITY}/{OOD_TIMESTAMP}`, or if normalized, then to `data/mahalanobis_scores/{OOD_WEATHER}/normalized/{OOD_SEVERITY}/{OOD_TIMESTAMP}`
> [!IMPORTANT]
> - The `NORMALIZATION`` variable defaults to false. Setting it to true applies Global Average Pooling and L2 Normalization to the feature vectors, and automatically routes all saved files to a dedicated `normalized/` sub-folder to prevent overwriting your raw data.
> - The `OOD_WEATHER` and `OOD_SEVERITY` tells BEVFormer which folder to target (Case sensitive).
> - The `OOD_WEATHER` can be `Fog` or `Snow` (for corrupted dataset), or `Clear` for clean dataset.
> - The `OOD_SEVERITY` can be `easy`, `mid`, or `hard` (for corrupted dataset) or `baseline` for clean dataset.
> - For `OOD_TIMESTAMP`, please check what timestap you will use from `data/intercepted_feature_logits/{target_folder}/{timestamp}`
> - For `BASELINE_TIMESTAMP`, please check what timestamp is used under `data/mahalanobis_distances/nuscenes/{timestamp}`
### Overall Folder Structure
```
data/
└── mahalanobis_distances/
    ├── nuscenes/
    │   ├── 1111/
    │   │   ├── mahalanobis_baseline.pt               <-- (Raw Math)
    │   │   ├── mahalanobis_baseline_normalized.pt    <-- (Normalized Math)
    │   │   └── mahalanobis_distances.json            <-- (Raw Scores)
    │   └── normalized/
    │       └── 1111/
    │           └── mahalanobis_distances.json        <-- (Normalized Scores)
    │
    └── Fog/
        └── easy/
            ├── 2222/
            │   └── mahalanobis_distances.json        <-- (Raw Scores)
            └── normalized/
                └── 2222/
                    └── mahalanobis_distances.json    <-- (Normalized Scores)
```
## AUROC & FPR95 Evaluations
### AUROC Evaluation
To get AUROC scores and generate ROC curve plots, run the following command:
```bash
# Raw Evaluation
OOD_WEATHER=Weather OOD_SEVERITY=severity OOD_TIMESTAMP=DATE_TIME BASELINE_TIMESTAMP=DATE_TIME python safety_monitor/auroc_evaluator.py

# Normalized Evaluation
NORMALIZATION=true OOD_WEATHER=Weather OOD_SEVERITY=severity OOD_TIMESTAMP=DATE_TIME BASELINE_TIMESTAMP=DATE_TIME python safety_monitor/auroc_evaluator.py
```
This will output the final evaluated `.png` plots to `plots/auroc/{OOD_WEATHER}/{OOD_SEVERITY}/{OOD_TIMESTAMP}` (or the `normalized/` subdirectory).
> [!IMPORTANT]
> - Adding `NORMALIZATION=true` will automatically update the title of your generated .png graph to say "(Normalized)" and will load the Mahalanobis JSON scores from your `normalized/` directories.
> - The `OOD_WEATHER` and `OOD_SEVERITY` tells BEVFormer which folder to target (Case sensitive).
> - The `OOD_WEATHER` can be `Fog` or `Snow` (for corrupted dataset).
> - The `OOD_SEVERITY` can be `easy`, `mid`, or `hard` (for corrupted dataset).
> - For `OOD_TIMESTAMP`, please check what timestap you will use from `data/intercepted_feature_logits/{target_folder}/{timestamp}`
> - For `BASELINE_TIMESTAMP`, please check what timestamp is used under `data/mahalanobis_distances/nuscenes/{timestamp}`
### FPR_95 Evaluation
To run get FPR95 scores, run the following command:
```bash
# Raw Evaluation
OOD_WEATHER=Weather OOD_SEVERITY=severity OOD_TIMESTAMP=DATE_TIME BASELINE_TIMESTAMP=DATE_TIME python safety_monitor/fpr95_evaluator.py

# Normalized Evaluation
NORMALIZATION=true OOD_WEATHER=Weather OOD_SEVERITY=severity OOD_TIMESTAMP=DATE_TIME BASELINE_TIMESTAMP=DATE_TIME python safety_monitor/fpr95_evaluator.py
```
This will output the final evaluated `.png` plots to `plots/fpr95/{OOD_WEATHER}/{OOD_SEVERITY}/{OOD_TIMESTAMP}` (or the `normalized/` subdirectory).
> [!IMPORTANT]
> - Adding `NORMALIZATION=true` will automatically update the title of your generated .png graph to say "(Normalized)" and will load the Mahalanobis JSON scores from your `normalized/` directories.
> - The `OOD_WEATHER` and `OOD_SEVERITY` tells BEVFormer which folder to target (Case sensitive).
> - The `OOD_WEATHER` can be `Fog` or `Snow`.
> - The `OOD_SEVERITY` can be `easy`, `mid`, or `hard` (for corrupted dataset).
>  - For `OOD_TIMESTAMP`, please check what timestap you will use from `data/intercepted_feature_logits/{target_folder}/{timestamp}`
> - For `BASELINE_TIMESTAMP`, please check what timestamp is used under `data/mahalanobis_distances/nuscenes/{timestamp}`

## Apptainer Workflow
This project uses a fully independent, containerized environment to ensure 100% reproducibility across different compute nodes. The container securely houses all core models (BEVFormer, mmdetection3d) and evaluation scripts.

If you do not have the `bevformer_native.sif` in your root directory, run the following:
```bash
sbatch build.sh
```
> [!WARNING]
> Running this will also delete any previous instances of `bevformer_native.sif` in the root.

This creates a `bevformer_native.sif` file in your root, which is essentially the container that holds our evaluation artifact.

### Understanding the `run_pipeline.sh`
Before submitting the job, open run_pipeline.sh and update the following variables to match your specific environment:
- Slurm Settings: Update `#SBATCH --account=NAISS...` with your active compute allocation ID, and adjust the `--gpus-per-node` if needed.
- Absolute Paths: Update `PROJECT_DIR` and `CONTAINER` to point to your exact directories.
- Experiment Toggles: Set your target `OOD_WEATHER` (e.g., Fog, Snow, Clear) and `OOD_SEVERITY`.
- Baseline ID: Update the `BASELINE_ID` variable with the timestamp of your clean baseline run. (This is strictly required for the Mahalanobis Distance calculations in Steps 3 and 4).
### Run pipeline:
From the root, run the following:
```bash
sbatch run_pipeline.sh
```
### Monitor the job
Because this runs in the background on compute nodes, it will not print to your terminal.

Check your job status and find your JOBID:
```bash
squeue -u $USER
```
Once the job state changes to R (Running), Slurm will generate a log file. You can watch the live output of your pipeline by running:
```bash
tail -f slurm-<JOBID>.out
```
