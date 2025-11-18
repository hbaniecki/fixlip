#!/bin/bash

#SBATCH --job-name=tbd
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH --output=sbatch_logs/log_%j.log
#SBATCH --time=01-23:50:00

set -e
hostname; pwd; date

module load anaconda/4.0
source $CONDA_SOURCE
conda activate fixlip

date

python insertion_deletion.py \
                  --model_name $1 \
                  --path_input $2 \
                  --path_output $3 \
                  --start $4 \
                  --stop $5

date
