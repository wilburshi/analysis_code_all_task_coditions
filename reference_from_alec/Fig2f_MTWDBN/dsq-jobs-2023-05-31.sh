#!/bin/bash
#SBATCH --output /dev/null
#SBATCH --array 0-8399
#SBATCH --job-name dsq-jobs
#SBATCH -p pi_jadi,day,scavenge --mail-type ALL --mem-per-cpu 8G -t 13:00:00 --requeue

# DO NOT EDIT LINE BELOW
/vast/palmer/apps/avx2/software/dSQ/1.05/dSQBatch.py --job-file /vast/palmer/home.mccleary/ags72/Documents/MTWDBN/Final_Figures/Synthetic/jobs.txt --status-dir /vast/palmer/home.mccleary/ags72/Documents/MTWDBN/Final_Figures/Synthetic

