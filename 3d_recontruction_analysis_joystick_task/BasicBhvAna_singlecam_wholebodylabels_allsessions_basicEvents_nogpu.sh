#!/bin/bash
#SBATCH --partition=pi_jadi
#SBATCH --job-name=BasicBhvAna_singlecam_wholebodylabels_allsessions_basicEvents_nogpu
#SBATCH --out="slurm-%j.out"
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=48000
#SBATCH --time=168:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weikang.shi@yale.edu

module purge
module load miniconda
conda activate DEEPLABCUT_YCRC
python BasicBhvAna_singlecam_wholebodylabels_allsessions_basicEvents.py
