#!/bin/bash
#SBATCH -J gp_training
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # Allocate 2 CPUs for process and I/O overhead
#SBATCH -t 06:00:00               # 12 hours wall-clock time
#SBATCH -o logs/slurm-%j.out      # Standard output log (logs/slurm-JOBID.out)
#SBATCH -e logs/slurm-%j.err      # Standard error log

# Ensure logs directory exists
mkdir -p logs

# Navigate to the project directory
cd /home/beizl42/projects/nanoLM

# Load environment
module load cuda/12.4.0
module load miniforge
conda activate nanolm

python train.py config/train_shakespeare.py