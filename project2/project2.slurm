#!/usr/bin/env bash

#SBATCH --job-name=4nodes
#SBATCH --partition=short-96core-shared
#SBATCH --output=36processor.txt
#SBATCH --ntasks-per-node=64
#SBATCH --nodes=1
#SBATCH --time=25:00

eval "$(conda shell.bash hook)"
conda activate my_parallel_env

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_NUM_THREADS=1

mpirun -np 64 python -u /gpfs/scratch/jmoylan/AMS-530-Principles-of-Parallel-Computing/project2/optimization.py
