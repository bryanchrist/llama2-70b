#!/bin/bash
#SBATCH -A bryan_research
##SBATCH -A ds5559
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:2 # replace X with the number of GPUs per node
#SBATCH -C gpupod

##SBATCH --hint=nomultithread #8GB per reserved CPU core if hyperthreading deactivated and 1/2 that when active
#SBATCH --cpus-per-task=10
#SBATCH --mem=250G
##SBATCH --gres=gpu:1
##SBATCH --gres=gpu:v100:1
##SBATCH --gres=gpu:a100_80gb:1
##SBATCH -c 1
#SBATCH -t 3-00:00:00
#SBATCH -J mathwell_topic
#SBATCH -o mathwell_topic-%A.out
#SBATCH -e mathwell_topic-%A.err
##SBATCH -x udc-an28-[1,7],udc-an34-[1,7,13,19]#,udc-an36-[1,13,19]#,udc-an37-[1,7,13,19]#,udc-ba25-2[3,7,8],udc-ba26-2[3-6],udc-ba27-2[3-4]

module purge

#module load singularity pytorch/1.8.1  # 2
#singularity run --nv $CONTAINERDIR/pytorch-1.8.1.sif finetune/adapter.py # 3

#module load singularity
#singularity run --nv pytorch_23.03-py3.sif python finetune/adapter.py 

module purge
module load anaconda
DIR=~/.conda/envs/mathwell
source activate mathwell
export PATH=$DIR/bin:$PATH
export LD_LIBRARY_PATH=$DIR/lib:$PATH
export PYTHONPATH=$DIR/lib/python3.11/site-packages:$PATH
export HF_HOME='/scratch/brc4cb/llama/cache'
python mathwell.py