#!/bin/bash
#SBATCH -p gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=70:00:00
#SBATCH --output=classify_prokbert_task3_%j.out
#SBATCH --error=classify_prokbert_task3_%j.err

module load anaconda3
conda activate dna_no_torch

# Ensure required packages
conda install -y scikit-learn pandas numpy psutil

# Run ProkBERT lifestyle classification for Task 3
python /work/sgk270/prokbert_task3_new/prokbert_task3_3_classify.py \
    --output_dir /work/sgk270/prokbert_task3_new/lifestyle_output \
    --num_train_epochs 500 \
    --early_stop_patience 15
