#!/bin/bash
#SBATCH --job-name=balance_task1_4
#SBATCH --output=balance_%j.out
#SBATCH --error=balance_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=compute1

echo "Job started at: $(date)"
echo "Running on: $(hostname)"

module purge
module load anaconda3
conda activate dna

POSITIVE_DIR="/work/sgk270/dataset_for_benchmarking/combine3/final_dataset"
NEGATIVE_DIR="/work/sgk270/dnabert2_task1_new/filtered_negative"
OUTPUT_DIR="/work/sgk270/dnabert2_task1_new"

python dnabert2_task1_4_balance.py \
  --positive "$POSITIVE_DIR" \
  --negative "$NEGATIVE_DIR" \
  --pos_pattern "*.fasta" \
  --neg_pattern "filtered_*.fna.gz" \
  --output_dir "$OUTPUT_DIR"

echo "Job finished at: $(date)"
