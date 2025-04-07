#!/bin/bash
#SBATCH -p gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=70:00:00
#SBATCH --output=embed_task3_genalm_%j.out
#SBATCH --error=embed_task3_genalm_%j.err

module load anaconda3
conda activate dna_no_torch

# Optional: ensure dependencies are installed
conda install -y psutil pandas numpy

# Run the GenaLM embedding script for Task 3
python3 /work/sgk270/genalm_task3_new/genalm_task3_1_embed.py \
  --model_name_or_path AIRI-Institute/gena-lm-bert-base-t2t-multi \
  --contigs_path /work/sgk270/dnabert2_task3_new/contigs \
  --full_seqs_path /work/sgk270/dnabert2_task3_new/full_seqs \
  --output_dir /work/sgk270/genalm_task3_new/contig_embed \
  --full_output_dir /work/sgk270/genalm_task3_new/full_embed \
  --batch_size 32
