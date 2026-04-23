#!/bin/bash
#SBATCH --job-name=Speed
#SBATCH --account=def-yangw-ab
#SBATCH --gpus-per-node=1
#SBATCH --mem=100000M               # memory per node
#SBATCH --time=2-00:00
#SBATCH --mail-user=ahmed_hani_dawoud@hotmail.com
#SBATCH --mail-type=ALL

# --- Activate Virtual Environment ---
echo "Activating python virtual environment"
source ~/py312/bin/activate

# --- Navigate to Project Directory and Run Script
echo "Changing to the project directory..."
cd /home/o7ahmed/projects/def-yangw-ab/o7ahmed/p-criaq/p-criaq/DA/

wandb offline
python domainadaptation_speed_5.py

