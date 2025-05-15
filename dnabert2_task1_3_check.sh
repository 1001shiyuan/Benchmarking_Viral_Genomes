#!/bin/bash
#SBATCH --job-name=task1_3_stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --partition=compute1
#SBATCH --output=check_%j.out
#SBATCH --error=check_%j.err

echo "Job started at: $(date)"
echo "Running on: $(hostname)"

module purge
module load anaconda3
conda activate dna

# Define positive and negative directories
POSITIVE_DIR="/work/sgk270/dataset_for_benchmarking/combine3/final_dataset"
NEGATIVE_DIR="/work/sgk270/dnabert2_task1_new/filtered_negative"

# Run parallel stats script
python dnabert2_task1_3_check.py \
  --positive "$POSITIVE_DIR" \
  --negative "$NEGATIVE_DIR" \
  --pos_pattern "*.fasta" \
  --neg_pattern "filtered_*.fna.gz"

echo "Job finished at: $(date)"
