#!/bin/bash
# Script to set up virtual environment for ProkBERT

# Load anaconda
module load anaconda3

# Create a new conda environment for ProkBERT
conda create -n prokbert python=3.8 -y

# Activate the environment
conda activate prokbert

# Install necessary packages
pip install torch transformers pandas numpy scikit-learn psutil