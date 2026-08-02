#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --account=naiss2026-4-688-gpu

echo "Starting ARM64 container build on compute node..."
rm -f bevformer_native.sif

apptainer build --fakeroot --ignore-fakeroot-command bevformer_native.sif bevformer_native.def

echo "Build complete!"
