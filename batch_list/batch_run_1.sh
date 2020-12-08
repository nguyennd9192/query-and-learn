#!/bin/bash 
#SBATCH --ntasks=2
#SBATCH --output=./output_1.txt
#SBATCH --cpus-per-task=16
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/code/sh/uniform_u_gp_mt_LFDA.sh
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/code/sh/uniform_u_gp_mt_LMNN.sh
