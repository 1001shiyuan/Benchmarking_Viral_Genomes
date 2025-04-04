#!/bin/bash
#SBATCH --job-name=prokbert_filter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --partition=compute1
#SBATCH --output=prokbert_filter_%j.out
#SBATCH --error=prokbert_filter_%j.err

# Load anaconda module
module load anaconda3

# Activate the prokbert environment
source activate prokbert

# Create output directories if they don't exist
mkdir -p /work/sgk270/prokbert_task2/order_full_embed
mkdir -p /work/sgk270/prokbert_task2/family_full_embed
mkdir -p /work/sgk270/prokbert_task2/genus_full_embed
mkdir -p /work/sgk270/prokbert_task2/order_contig_embed
mkdir -p /work/sgk270/prokbert_task2/family_contig_embed
mkdir -p /work/sgk270/prokbert_task2/genus_contig_embed

# Run the filtering script
cd /work/sgk270/prokbert_task2
python filter.py

echo "Filtering completed!"