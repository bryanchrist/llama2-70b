#!/bin/bash
#SBATCH -A bryan_research
#SBATCH --partition=standard
#SBATCH --ntasks=1

#SBATCH -t 4-00:00:00
#SBATCH -J gpt4_solvability
#SBATCH -o gpt4_solvability-%j.out
#SBATCH -e gpt4_solvability-%j.err

module purge
module load anaconda
DIR=~/.conda/envs/mathwell
source activate mathwell
export PATH=$DIR/bin:$PATH
export LD_LIBRARY_PATH=$DIR/lib:$PATH
export PYTHONPATH=$DIR/lib/python3.11/site-packages:$PATH
python gpt4_solvability.py