#!/bin/bash
#SBATCH --job-name=refusal_Q2
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j_Q2.log
#SBATCH --partition=long

##SBATCH --partition=debug

mkdir -p logs


MODEL_PATH="/data/users/gabben/TDA602/models/qwen2/"

# Stage 1+2 only: direction finding, no generation or eval
apptainer exec --nv ./pytorch_2.3.1-cuda12.1-cudnn8-devel.sif \
    bash -c "source /data/users/gabben/TDA602/refusal_direction/venv/bin/activate && \
python -m pipeline.run_pipeline2 \
    --model_path "Qwen/Qwen2-7B-Instruct" \
    --n_train 128 --n_val 32 --n_test 100 \
    --ce_loss_n_batches 32 --ce_loss_batch_size 4 \
    --max_new_tokens 512
"
#   --skip_filter \
#    --skip_completions \
#    --skip_eval \
#   --skip_loss \
    
echo "Job finished with exit code $?"
