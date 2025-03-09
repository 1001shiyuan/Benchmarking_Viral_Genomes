#!/bin/bash
#SBATCH --job-name=balance_data
#SBATCH --output=balance_data_%j.out
#SBATCH --error=balance_data_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=compute1

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

# Load necessary modules
module purge
module load anaconda3

# Activate virtual environment
source bert1/bin/activate

# Run the Python script with our parameters
python 4_balance.py \
    --positive "/work/sgk270/dataset_for_benchmarking/identification" \
    --negative "/work/sgk270/dnabert2_task1/filtered_negative" \
    --pos_pattern "*.fasta" \
    --neg_pattern "filtered_*.fna.gz" \
    --output_dir "/work/sgk270/dnabert2_task1"

echo "Job finished at: $(date)"