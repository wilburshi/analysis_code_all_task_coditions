#!/bin/bash
#SBATCH --partition=gpu_devel
#SBATCH --job-name=aniposelib_3d_reconstruction
#SBATCH --out="slurm-%j.out"
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=5200
#SBATCH --time=0-12
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weikang.shi@yale.edu
#SBATCH --gpus=1

module purge
module load cuDNN
module load Xvfb

module load miniconda
conda activate DLC

xvfb-run python aniposelib_3d_reconstruction.py
