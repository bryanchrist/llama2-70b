#!/bin/bash
#SBATCH -A bryan_research
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH -t 2-00:00:00
#SBATCH -J perplexity
#SBATCH -o perplexity-%A.out
#SBATCH -e perplexity-%A.err

module purge
module load anaconda
DIR=~/.conda/envs/mathwell
source activate mathwell
export PATH=$DIR/bin:$PATH
export LD_LIBRARY_PATH=$DIR/lib:$PATH
export PYTHONPATH=$DIR/lib/python3.11/site-packages:$PATH
export HF_HOME='/scratch/brc4cb/phi3/cache'
python perplexity.py