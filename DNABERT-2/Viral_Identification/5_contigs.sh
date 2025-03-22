#!/bin/bash
#SBATCH -p gpu2v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=70:00:00
#SBATCH --output=contigs_%j.out
#SBATCH --error=contigs_%j.err

# Load Anaconda and activate environment
module load anaconda3
conda activate dna_no_torch

# Install Biopython if not already installed
conda install -y biopython numpy

# Run the script
python3 5_contigs.py