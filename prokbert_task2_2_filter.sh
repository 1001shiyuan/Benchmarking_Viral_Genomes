#!/bin/bash
#SBATCH --job-name=prokbert_filter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --partition=compute1
#SBATCH --output=filter_%j.out
#SBATCH --error=filter_%j.err

module load anaconda3
conda activate dna_no_torch

# Ensure the updated C++ runtime is available
conda install -y -c conda-forge libstdcxx-ng

# Get conda lib path and set LD_LIBRARY_PATH
CONDA_LIB=$(python -c "import os, sys; print(os.path.join(sys.prefix, 'lib'))")
export LD_LIBRARY_PATH=$CONDA_LIB:$LD_LIBRARY_PATH

# Run the filtering script
python /work/sgk270/prokbert_task2_new/prokbert_task2_2_filter.py

