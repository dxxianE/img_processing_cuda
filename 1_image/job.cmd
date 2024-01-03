#!/bin/bash

#SBATCH --job-name=image
#SBATCH --output=output.out
#SBATCH --error=output.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1             	# Number of gpus
#SBATCH --time=00:10:00

module load CUDA
module load NVHPC

make || exit 1
./imgproc lleo 1920 1214

