#!/bin/bash
#SBATCH --job-name=neg_download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --partition=compute1
#SBATCH --output=neg_download_%j.out
#SBATCH --error=neg_download_%j.err

# Load modules
module load anaconda3

# Activate environment
conda activate dna
module load parallel

# genome updater
wget --quiet --show-progress https://raw.githubusercontent.com/pirovc/genome_updater/master/genome_updater.sh
chmod +x genome_updater.sh

# Run the Python script
python dnabert2_task1_1_download_data.py
