#!/bin/bash
#SBATCH -p gpu2v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=70:00:00
#SBATCH --output=contigs_%j.out
#SBATCH --error=contigs_%j.err
module load anaconda3
conda activate dna_no_torch
python3 dnabert2_task3_1_generate_contigs.py