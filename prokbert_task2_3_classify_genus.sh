#!/bin/bash
#SBATCH -p gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=70:00:00
#SBATCH --output=classify_genus_%j.out
#SBATCH --error=classify_genus_%j.err

module load anaconda3
conda activate dna_no_torch

# Optional: ensure dependencies are installed
conda install -y scikit-learn psutil pandas numpy

# Run ProkBERT Genus-level classification
python /work/sgk270/prokbert_task2_new/prokbert_task2_3_classify_genus.py
