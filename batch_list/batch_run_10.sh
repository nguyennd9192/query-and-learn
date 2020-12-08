#!/bin/bash 
#SBATCH --ntasks=2
#SBATCH --output=./output_10.txt
#SBATCH --cpus-per-task=16
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/code/sh/margin_u_gp_None.sh
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/code/sh/margin_u_gp_MLKR.sh
