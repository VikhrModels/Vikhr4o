#!/bin/bash
#SBATCH --job-name=salt                  # Job name
#SBATCH --output=output_%j.log           # Standard output and error log
#SBATCH --ntasks=1                       # Number of tasks (usually 1 for a single job)
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --nodelist=gpu80-3
#SBATCH --cpus-per-task=6                # Number of CPU cores per task
#SBATCH --partition=nlp-dept             # Partition name
#SBATCH --qos=nlp-pool
#SBATCH --time=48:00:00                  # Time limit (HH:MM:SS)

conda activate 4o

# Execute your command using srun
bash scripts/run_me.sh
