#!/bin/bash
#SBATCH -A bryan_research
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:2 # replace X with the number of GPUs per node
#SBATCH -C gpupod

##SBATCH --hint=nomultithread #8GB per reserved CPU core if hyperthreading deactivated and 1/2 that when active
#SBATCH --cpus-per-task=10
#SBATCH --mem=500G
##SBATCH --gres=gpu:1
##SBATCH --gres=gpu:v100:1
##SBATCH --gres=gpu:a100_80gb:1
##SBATCH -c 1
#SBATCH -t 3-00:00:00
#SBATCH -J llama_70b__finetune
#SBATCH -o llama_final_finetune-%A.out
#SBATCH -e llama_final_finetune-%A.err
##SBATCH -x udc-an28-[1,7],udc-an34-[1,7,13,19]#,udc-an36-[1,13,19]#,udc-an37-[1,7,13,19]#,udc-ba25-2[3,7,8],udc-ba26-2[3-6],udc-ba27-2[3-4]

module purge

#module load singularity pytorch/1.8.1  # 2
#singularity run --nv $CONTAINERDIR/pytorch-1.8.1.sif finetune/adapter.py # 3

#module load singularity
#singularity run --nv pytorch_23.03-py3.sif python finetune/adapter.py 

module load anaconda
module load cuda/11.4.2
source activate falcon_40B
conda install -n falcon_40B python-dotenv
pip install -U -r requirements.txt
conda install -y cudatoolkit
export LD_LIBRARY_PATH='/opt/conda/lib/' 
python qlora_no_embed.py \
    --model_name_or_path meta-llama/Llama-2-70b-hf \
    --output_dir ./llama_finetuned \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 250 \
    --save_total_limit 40 \
    --evaluation_strategy steps \
    --eval_dataset_size 1500 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 400 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset="train.json" \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 5000 \
    --eval_steps 187 \
    --learning_rate 0.0001 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --weight_decay 0.0 \
    --seed 0
