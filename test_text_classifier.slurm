#!/bin/bash

#SBATCH -A sds-phd-2022
#SBATCH -p gpu

#SBATCH --gres=gpu:a100:4 # replace X with the number of GPUs per node
#SBATCH -C gpupod

#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=9000
##SBATCH --mem=80G
#SBATCH -t 3-00:00:00
#SBATCH -J test_text_classifier
#SBATCH -o test_text_classifier-%A.out
#SBATCH -e test_text_classifier-%A.err

module purge

unset LD_LIBRARY_PATH
module load anaconda
#module load cuda/11.4.2
source activate falcon_40B
python text_classifier.py
