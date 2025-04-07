#!/bin/bash
#SBATCH -p gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=70:00:00
#SBATCH --output=classify_order_%j.out
#SBATCH --error=classify_order_%j.err

module load anaconda3
conda activate dna_no_torch

# Ensure dependencies are installed (optional if already handled in environment)
conda install -y scikit-learn psutil pandas numpy

# Run classification for Order level
python dnabert2_task2_4_classify_order.py \
    --label_type order \
    --output_dir order_output \
    --num_train_epochs 500 \
    --early_stop_patience 15
