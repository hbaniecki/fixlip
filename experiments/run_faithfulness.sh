#!/bin/bash

#SBATCH --job-name=tbd
#SBATCH --gpus=2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=48G
#SBATCH --output=sbatch_logs/log_%j.log
#SBATCH --time=01-23:55:00
#SBATCH --nodes=1

set -e
hostname; pwd; date

module load anaconda/4.0
source $CONDA_SOURCE
conda activate fixlip

date

# python faithfulness.py \
#     --model_name openai/clip-vit-base-patch32 \
#     --path_input ../results/mscoco \
#     --path_output ../results \
#     --start 0 \
#     --stop 1000
python faithfulness.py \
    --model_name openai/clip-vit-base-patch16 \
    --path_input ../results/mscoco \
    --path_output ../results \
    --start 0 \
    --stop 1000

date
