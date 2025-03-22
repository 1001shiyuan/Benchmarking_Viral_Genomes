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

# Install required packages if not present
conda install -y scikit-learn psutil pandas numpy

# Run script normally, letting SLURM handle output and errors
python3 classify_order.py
