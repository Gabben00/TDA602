#!/bin/bash
#SBATCH --job-name=refusal_debug
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j_debug.log

##SBATCH --partition=debug

mkdir -p logs


MODEL_PATH="/data/users/gabben/TDA602/models/qwen2/"

# Stage 1+2 only: direction finding, no generation or eval
apptainer exec --nv ./pytorch_2.3.1-cuda12.1-cudnn8-devel.sif \
    bash -c "source /data/users/gabben/TDA602/refusal_direction/venv/bin/activate && \
python -m pipeline.run_pipeline_debug \
    --model_path "Qwen/Qwen3-8B" \
    --n_train 8 --n_val 4 --n_test 4 \
    --skip_filter \
    --max_new_tokens 20
"
#    --skip_completions \
#    --skip_eval \
#    --skip_loss \
    
echo "Job finished with exit code $?"
