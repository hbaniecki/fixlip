#!/bin/bash

#SBATCH --job-name=tbd
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH --output=sbatch_logs/log_%j.log
#SBATCH --time=00-23:00:00

set -e
hostname; pwd; date

module load anaconda/4.0
source $CONDA_SOURCE
conda activate fixlip

date

python pointing_game_shapley.py --model_name $1 \
                                --path_input $2 \
                                --path_output $3 \
                                --mode $4 \
                                --class_labels $5 \
                                --budget $6 \
                                --batch_size $7 \
                                --random_state $8

date
