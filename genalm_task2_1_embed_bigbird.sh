#!/bin/bash
#SBATCH -p gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=genalm_embed_%j.out
#SBATCH --error=genalm_embed_%j.err

# Load environment
module load anaconda3
source activate genalm

# Huggingface cache directory
export HF_HOME=/work/sgk270/genalm_task2_new/.cache/huggingface
export CUDA_LAUNCH_BLOCKING=1

# Print environment info
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"

# Run script
python /work/sgk270/genalm_task2_new/genalm_task2_1_embed_bigbird.py \
  --model_name_or_path AIRI-Institute/gena-lm-bert-base-t2t-multi \
  --contigs_path /work/sgk270/dnabert2_task2_new/contigs \
  --full_seqs_path /work/sgk270/dnabert2_task2_new/full_seqs \
  --output_dir /work/sgk270/genalm_task2_new/contig_embed \
  --full_output_dir /work/sgk270/genalm_task2_new/full_embed \
  --batch_size 32
