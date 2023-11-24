#!/bin/bash -l
# SBATCH --time=72:00:00
# SBATCH --mem=512G

# Run your code here
module load anaconda
python task1_training_scripts.py
