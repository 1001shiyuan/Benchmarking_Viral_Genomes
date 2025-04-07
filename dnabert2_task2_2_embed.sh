#!/bin/bash
#SBATCH -p gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=70:00:00
#SBATCH --output=embed_%j.out
#SBATCH --error=embed_%j.err
module load anaconda3
conda activate dna
# Install required packages if not present
conda install -y psutil pandas numpy
# Run precompute script
python3 dnabert2_task2_2_embed.py