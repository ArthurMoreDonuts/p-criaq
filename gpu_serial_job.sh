#!/bin/bash
#SBATCH --account=def-yangw-ab
#SBATCH --gpus-per-node=1
#SBATCH --mem=16000M               # memory per node
#SBATCH --time=0-12:00

python main.py