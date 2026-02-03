#!/bin/bash
#SBATCH --partition=week
#SBATCH --job-name=BasicBhvAna_and_ContVariAna_Aniposelib3d_allsessions_basicEvents_autolever_nogpu
#SBATCH --out="slurm-%j.out"
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=48000
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weikang.shi@yale.edu

module purge
module load miniconda
conda activate DLC
python BasicBhvAna_and_ContVariAna_Aniposelib3d_allsessions_basicEvents_autolever.py
