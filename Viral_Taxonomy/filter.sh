#!/bin/bash
#SBATCH --job-name=filter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --time=24:00:00
#SBATCH --partition=compute1
#SBATCH --output=filter_%j.out
#SBATCH --error=filter_%j.err

python filter.py