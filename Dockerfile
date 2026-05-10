FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    wget \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.8 /usr/bin/python

RUN python -m pip install --upgrade pip
RUN pip install setuptools==59.5.0 wheel

RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 --extra-index-url https://download.pytorch.org/whl/cu111
RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

RUN pip install mmdet==2.14.0 mmsegmentation==0.14.1

RUN pip install trimesh==2.35.39 tensorboard==2.11.0 scikit-image==0.19.3

ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"
ENV FORCE_CUDA="1"
RUN git clone https://github.com/open-mmlab/mmdetection3d.git /mmdetection3d && \
    cd /mmdetection3d && \
    git checkout v0.17.1 && \
    pip install -e . --no-deps

RUN pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13 pylint ipython==8.12 numba==0.48.0 pandas==1.4.4 pyquaternion shapely fire cachetools scikit-learn

RUN pip install https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/detectron2-0.6%2Bcu111-cp38-cp38-linux_x86_64.whl
RUN pip install lyft_dataset_sdk nuscenes-devkit plyfile networkx==2.2

RUN pip install numpy==1.19.5 matplotlib==3.5.2 typing-extensions==4.5.0 Pillow==9.5.0 setuptools==59.5.0 pyquaternion shapely fire cachetools scikit-learn

WORKDIR /workspace
