#!/bin/bash
#SBATCH --job-name=Altitude
#SBATCH --account=def-yangw-ab
#SBATCH --gpus-per-node=1
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=02:00:00
#SBATCH --mail-user=ahmed_hani_dawoud@hotmail.com
#SBATCH --mail-type=ALL

# --- Activate Virtual Environment ---
echo "Activating python virtual environment"
source ~/py312/bin/activate

# --- Navigate to Project Directory and Run Script
echo "Changing to the project directory..."
cd /home/o7ahmed/scratch/p-craiq
python altitude.py