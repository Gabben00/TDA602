#!/bin/bash
#SBATCH --job-name=pip_install
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=pip_install.log
##SBATCH --partition=debug

source /data/users/gabben/TDA602/refusal_direction/venv/bin/activate
pip install -r requirements.txt --upgrade