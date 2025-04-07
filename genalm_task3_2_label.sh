#!/bin/bash
#SBATCH --job-name=label_task3_genalm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=compute1
#SBATCH --output=label_task3_genalm_%j.out
#SBATCH --error=label_task3_genalm_%j.err

module load anaconda3
conda activate dna_no_torch

# Install newer libstdc++ if needed
conda install -y -c conda-forge libstdcxx-ng

# Get conda lib path
CONDA_LIB=$(python -c "import os, sys; print(os.path.join(sys.prefix, 'lib'))")

# Prepend to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_LIB:$LD_LIBRARY_PATH

# Run lifestyle label annotation
python /work/sgk270/genalm_task3_new/genalm_task3_2_label.py
