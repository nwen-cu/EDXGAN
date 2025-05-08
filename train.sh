#!/bin/bash
#SBATCH --job-name chroma_edx
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node v100s:1
#SBATCH --cpus-per-task 12
#SBATCH --mem 200gb
#SBATCH --time 336:00:00
#SBATCH --signal SIGUSR1@300
#SBATCH --partition zhaocmml

cd ~/lake-isee/nwen/arl_edx_chromagan/
ls

eval "$(micromamba shell hook --shell bash)"
micromamba activate dl

which python

srun --export=ALL python main128.py
