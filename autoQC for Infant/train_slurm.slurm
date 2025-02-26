#!/bin/bash
#SBATCH -J qc
#SBATCH -p bme_gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH -N 1
#SBATCH -t 3-0:00:00
#SBATCH --output=slurm_out/train.out
#SBATCH --error=slurm_out/train.err

hostname
date
nvidia-smi

python train_test_01.py

date
