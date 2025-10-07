#!/bin/bash
#SBATCH --account=def-yangw-ab
#SBATCH --gpus-per-node=1
#SBATCH --mem=1000M               # memory per node
#SBATCH --time=0-31:00
python main.py