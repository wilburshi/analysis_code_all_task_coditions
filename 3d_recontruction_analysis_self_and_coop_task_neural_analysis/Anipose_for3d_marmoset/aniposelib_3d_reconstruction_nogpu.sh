#!/bin/bash
#SBATCH --partition=week
#SBATCH --job-name=marmosets_aniposelib_3d_reconstruction_nogpu
#SBATCH --out="slurm-%j.out"
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=5200
#SBATCH --time=50:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weikang.shi@yale.edu

module purge
module load miniconda
conda activate DLC
python aniposelib_3d_reconstruction.py
