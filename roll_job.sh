#!/bin/bash
#SBATCH --account=def-yangw-ab
#SBATCH --gpus-per-node=1
#SBATCH --mem=16000M               # memory per node
#SBATCH --time=0-24:00
#SBATCH --mail-user=ahmed_hani_dawoud@hotmail.com
#SBATCH --mail-type=ALL
python roll.py