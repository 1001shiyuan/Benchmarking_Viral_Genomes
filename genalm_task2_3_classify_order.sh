#!/bin/bash
#SBATCH -p gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=70:00:00
#SBATCH --output=classify_order_%j.out
#SBATCH --error=classify_order_%j.err

# Load environment
module load anaconda3
conda activate dna_no_torch

# Install any missing dependencies (optional if already installed)
conda install -y scikit-learn pandas numpy

# Run GenaLM Order-level classification
python /work/sgk270/genalm_task2_new/genalm_task2_3_classify_order.py \
  --output_dir /work/sgk270/genalm_task2_new/order_output \
  --num_train_epochs 500 \
  --early_stop_patience 15
