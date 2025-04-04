#!/bin/bash
#SBATCH -p gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=70:00:00
#SBATCH --output=genalm_classify_family_%j.out
#SBATCH --error=genalm_classify_family_%j.err

# Load anaconda module
module load anaconda3

# Activate the genalm environment
source activate genalm

# Set working directory
cd /work/sgk270/genalm_task2

# Set cache directory for huggingface to avoid permission issues
export HF_HOME=/work/sgk270/genalm_task2/.cache/huggingface

# Create output directory
mkdir -p /work/sgk270/genalm_task2/family_output

# Run classification for family
python classify.py \
  --label_type family \
  --output_dir /work/sgk270/genalm_task2/family_output \
  --num_train_epochs 500 \
  --early_stop_patience 15 \
  --base_dir /work/sgk270/genalm_task2

echo "Family classification completed!"