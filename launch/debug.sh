#!/bin/bash

#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=03:30:00
#SBATCH --job-name=vscode-tunnelcd
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --output=/iopsstor/scratch/cscs/%u/slurmlogs/vscode-tunnle-debug.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/slurmlogs/vscode-tunnle-debug.err

srun --environment=pytorch_env \
 --container-mounts=$HOME/vscode-cli-$(arch)/code:/code \
 /code tunnel --accept-server-license-terms \
 --name=$CLUSTER_NAME-tunnelcd
