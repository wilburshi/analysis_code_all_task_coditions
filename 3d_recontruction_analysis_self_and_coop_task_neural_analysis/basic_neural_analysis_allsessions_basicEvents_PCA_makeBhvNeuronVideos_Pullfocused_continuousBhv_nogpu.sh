#!/bin/bash
#SBATCH --partition=week
#SBATCH --job-name=basic_neural_analysis_allsessions_basicEvents_PCA_makeBhvNeuronVideos_Pullfocused_continuousBhv_nogpu
#SBATCH --out="slurm-%j.out"
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=160000
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weikang.shi@yale.edu

module purge
module load miniconda
conda activate DLC
python basic_neural_analysis_allsessions_basicEvents_PCA_makeBhvNeuronVideos_Pullfocused_continuousBhv.py
