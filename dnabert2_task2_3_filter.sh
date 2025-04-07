#!/bin/bash
#SBATCH --job-name=filter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --partition=compute1
#SBATCH --output=filter_%j.out
#SBATCH --error=filter_%j.err
module load anaconda3
conda activate dna_no_torch

# Install the newer libstdc++ if not already installed
conda install -y -c conda-forge libstdcxx-ng

# Get the path to conda's lib directory
CONDA_LIB=$(python -c "import os, sys; print(os.path.join(sys.prefix, 'lib'))")

# Prepend the conda lib to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_LIB:$LD_LIBRARY_PATH

# Run the script
python dnabert2_task2_3_filter.py
