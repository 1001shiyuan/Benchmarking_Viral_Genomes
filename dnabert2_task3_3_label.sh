#!/bin/bash
#SBATCH --job-name=label_task3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=compute1
#SBATCH --output=label_task3_%j.out
#SBATCH --error=label_task3_%j.err

module load anaconda3
conda activate dna_no_torch

# Install newer libstdc++ if needed
conda install -y -c conda-forge libstdcxx-ng

# Get conda's lib path
CONDA_LIB=$(python -c "import os, sys; print(os.path.join(sys.prefix, 'lib'))")

# Prepend to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_LIB:$LD_LIBRARY_PATH

# Run the label script
python dnabert2_task3_3_label.py
