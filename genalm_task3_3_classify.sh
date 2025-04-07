#!/bin/bash
#SBATCH -p gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=70:00:00
#SBATCH --output=classify_genalm_task3_%j.out
#SBATCH --error=classify_genalm_task3_%j.err

module load anaconda3
conda activate dna_no_torch

# Optional: ensure required packages are available
conda install -y scikit-learn pandas numpy psutil

# Run GenALM lifestyle classification for Task 3
python genalm_task3_3_classify.py \
    --output_dir /work/sgk270/genalm_task3_new/lifestyle_output \
    --num_train_epochs 500 \
    --early_stop_patience 15
