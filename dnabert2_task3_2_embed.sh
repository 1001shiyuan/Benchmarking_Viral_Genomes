#!/bin/bash
#SBATCH -p gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=70:00:00
#SBATCH --output=embed_task3_%j.out
#SBATCH --error=embed_task3_%j.err

module load anaconda3
conda activate dna_no_torch

# Optional: ensure packages are available
conda install -y psutil pandas numpy

# Run embedding script
python3 dnabert2_task3_2_embed.py
