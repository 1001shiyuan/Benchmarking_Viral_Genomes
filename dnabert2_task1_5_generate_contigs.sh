#!/bin/bash
#SBATCH --job-name=task1_5_contigs
#SBATCH --output=generate_contigs_%j.out
#SBATCH --error=generate_contigs_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --partition=compute1

echo "Job started at: $(date)"
echo "Node: $(hostname)"

module purge
module load anaconda3
conda activate dna

python dnabert2_task1_5_generate_contigs.py

echo "Job finished at: $(date)"
