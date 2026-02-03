#!/bin/bash
#SBATCH --partition=pi_jadi
#SBATCH --job-name=marmosets_aniposelib_3d_reconstruction_mixedpairs_nogpu
#SBATCH --out="slurm-%j.out"
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=5120
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weikang.shi@yale.edu

module purge
module load miniconda
conda activate DEEPLABCUT_YCRC
python aniposelib_3d_reconstruction_mixedpairs.py
