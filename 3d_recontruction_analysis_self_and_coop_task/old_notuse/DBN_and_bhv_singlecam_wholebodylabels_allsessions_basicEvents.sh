#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=DBN_and_bhv_singlecam_wholebodylabels_allsessions_basicEvents
#SBATCH --out="slurm-%j.out"
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=160000
#SBATCH --time=0-12
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weikang.shi@yale.edu
#SBATCH --gpus=1

module purge
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load Xvfb/1.20.9-GCCcore-10.2.0

module load miniconda
conda activate DEEPLABCUT_YCRC

xvfb-run python DBN_and_bhv_singlecam_wholebodylabels_allsessions_basicEvents.py
