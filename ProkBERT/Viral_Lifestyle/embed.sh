#!/bin/bash
#SBATCH -p gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=prokbert_lifestyle_embed_%j.out
#SBATCH --error=prokbert_lifestyle_embed_%j.err

# Load anaconda module
module load anaconda3

# Activate the prokbert environment created for task2
source activate prokbert

# Create output directories if they don't exist
mkdir -p /work/sgk270/prokbert_task3/lifestyle_contig_embed
mkdir -p /work/sgk270/prokbert_task3/lifestyle_full_embed

# Set cache directory for huggingface to avoid permission issues
export HF_HOME=/work/sgk270/prokbert_task3/.cache/huggingface

# Add environment variable for better CUDA error reporting
export CUDA_LAUNCH_BLOCKING=1

# Print environment information
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"

# Run the embedding script
python /work/sgk270/prokbert_task3/embed.py \
  --model_name_or_path neuralbioinfo/prokbert-mini \
  --contigs_path /work/sgk270/dnabert2_task3/lifestyle_contigs \
  --full_seqs_path /work/sgk270/dnabert2_task3/lifestyle_full_seqs \
  --output_dir /work/sgk270/prokbert_task3/lifestyle_contig_embed \
  --full_output_dir /work/sgk270/prokbert_task3/lifestyle_full_embed \
  --batch_size 32