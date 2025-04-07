#!/bin/bash
#SBATCH -p gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=70:00:00
#SBATCH --output=classify_lifestyle_%j.out
#SBATCH --error=classify_lifestyle_%j.err

module load anaconda3
conda activate dna_no_torch

# Optional: ensure required packages are installed
conda install -y scikit-learn pandas numpy psutil

# Run classification
python dnabert2_task3_4_classify.py \
    --output_dir lifestyle_output \
    --num_train_epochs 500 \
    --early_stop_patience 15
