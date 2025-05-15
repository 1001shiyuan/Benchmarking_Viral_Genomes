#!/bin/bash
#SBATCH --job-name=filter_viral_db_then_array
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-59
#SBATCH --partition=compute2
#SBATCH --output=filter_viral_%A_%a.out
#SBATCH --error=filter_viral_%A_%a.err

module purge
module load anaconda3
conda activate dna
module load ncbi/blast/2.11.0

# Make sure Biopython is available
conda list | grep biopython || conda install -y biopython

python dnabert2_task1_2_filter_viral_elements.py