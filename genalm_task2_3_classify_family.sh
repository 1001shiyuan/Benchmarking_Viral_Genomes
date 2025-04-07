#!/bin/bash
#SBATCH -p gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=70:00:00
#SBATCH --output=classify_family_%j.out
#SBATCH --error=classify_family_%j.err

module load anaconda3
conda activate dna_no_torch

# Optional: ensure dependencies are installed
conda install -y scikit-learn psutil pandas numpy

# Run GenaLM Family-level classification
python /work/sgk270/genalm_task2_new/genalm_task2_3_classify_family.py
