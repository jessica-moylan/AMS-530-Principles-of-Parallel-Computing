#!/usr/bin/env bash

#SBATCH --job-name=project2
#SBATCH --partition=short-40core-shared
#SBATCH --output=output.txt
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --time=10:00

# 3. Initialize conda hooks for non-interactive Slurm shell sessions
eval "$(conda shell.bash hook)"
# 4. Activate your environment
conda activate my_parallel_env

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_NUM_THREADS=1

# 5. Run python using mpirun
mpirun -np 16 python -u  /gpfs/scratch/jmoylan/AMS-530-Principles-of-Parallel-Computing/project2/optimization.py
