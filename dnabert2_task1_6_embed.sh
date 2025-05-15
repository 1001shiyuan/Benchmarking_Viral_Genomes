#!/bin/bash
#SBATCH -p gpu2v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=70:00:00
#SBATCH --output=embed_task1_6_%j.out
#SBATCH --error=embed_task1_6_%j.err

module load anaconda3
conda activate dna

# Ensure dependencies are present
conda install -y psutil pandas numpy

# Run Task 1.6 embedding script
python3 dnabert2_task1_6_embed.py
