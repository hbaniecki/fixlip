#!/bin/bash

#SBATCH --job-name=tbd
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=62G
#SBATCH --output=sbatch_logs/log_%j.log
#SBATCH --time=00-23:50:00

set -e
hostname; pwd; date

module load anaconda/4.0
source $CONDA_SOURCE
conda activate fixlip

date

python explain_mscoco.py --model_name $1 \
                         --path_input $2 \
                         --path_output $3 \
                         --start $4 \
                         --stop $5 \
                         --mode $6 \
                         --p_sampler $7 \
                         --budget $8 \
                         --batch_size $9 \
                         --random_state ${10}

date
