#!/bin/bash
#SBATCH -p gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=70:00:00
#SBATCH --output=embed_task3_prokbert_%j.out
#SBATCH --error=embed_task3_prokbert_%j.err

module load anaconda3
conda activate dna_no_torch

# Optional: install dependencies
conda install -y psutil pandas numpy

# Run the ProkBERT embedding script for Task 3
python3 /work/sgk270/prokbert_task3_new/prokbert_task3_1_embed.py \
  --model_name_or_path neuralbioinfo/prokbert-mini \
  --contigs_path /work/sgk270/dnabert2_task3_new/contigs \
  --full_seqs_path /work/sgk270/dnabert2_task3_new/full_seqs \
  --output_dir /work/sgk270/prokbert_task3_new/contig_embed \
  --full_output_dir /work/sgk270/prokbert_task3_new/full_embed \
  --batch_size 32
