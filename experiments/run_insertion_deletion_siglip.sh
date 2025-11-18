#!/bin/bash

#SBATCH --job-name=tbd
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=48G
#SBATCH --output=sbatch_logs/log_%j.log
#SBATCH --time=00-23:50:00

set -e
hostname; pwd; date

module load anaconda/4.0
source $CONDA_SOURCE
conda activate fixlip

date

python insertion_deletion_siglip.py \
                  --model_name $1 \
                  --path_input $2 \
                  --path_output $3 \
                  --path_metadata $4 \
                  --mode $5 \
                  --p_sampler $6 \
                  --budget $7 

date