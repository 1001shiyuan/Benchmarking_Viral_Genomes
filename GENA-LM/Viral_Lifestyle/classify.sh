#!/bin/bash
#SBATCH -p gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=70:00:00
#SBATCH --output=genalm_classify_%j.out
#SBATCH --error=genalm_classify_%j.err

# Load anaconda module
module load anaconda3

# Activate the genalm environment
source activate genalm

# Set up output directory
mkdir -p /work/sgk270/genalm_task3/lifestyle_output

# Working directory
cd /work/sgk270/genalm_task3

# Run classification script
python classify.py \
  --num_train_epochs 500 \
  --output_dir /work/sgk270/genalm_task3/lifestyle_output \
  --early_stop_patience 15

echo "GENA-LM lifestyle classification completed!"