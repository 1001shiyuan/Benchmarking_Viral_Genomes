#!/bin/bash
# Script to set up virtual environment for GENA-LM

# Load anaconda
module load anaconda3

# Create a new conda environment for GENA-LM
conda create -n genalm python=3.8 -y

# Activate the environment
conda activate genalm

# Install necessary packages
pip install torch transformers pandas numpy scikit-learn psutil

# Print success message
echo "GENA-LM environment setup complete. To use it, run 'conda activate genalm'"