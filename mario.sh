#!/bin/bash
#
#SBATCH --job-name=MARIO
#SBATCH --output=/zooper2/esther.whang/mario/output.txt
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1-1
#SBATCH --mem=30gb
source /zooper2/esther.whang/.bashrc
python -m diffwave /zooper2/esther.whang/mario/diffwave/models_old /zooper2/esther.whang/mario/resampled_dataset/combined --max_steps 100000
python -m diffwave /zooper2/esther.whang/mario/diffwave/models_old /zooper2/esther.whang/mario/resampled_dataset/combined --max_steps 100000