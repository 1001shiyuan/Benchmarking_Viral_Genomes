#!/bin/bash
#SBATCH --job-name=neg_download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --partition=compute1
#SBATCH --output=neg_download_%j.out
#SBATCH --error=neg_download_%j.err

# Load 'parallel'
module load parallel

# Run the Python script
python 1_download_data.py
