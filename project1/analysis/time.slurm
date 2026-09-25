#!/usr/bin/env bash

#SBATCH --job-name=project1
#SBATCH --partition=short-40core-shared
#SBATCH --output=timeanlysis40.txt
#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH --time=05:00

module load mpi4py/latest

mpirun -np 40 python /gpfs/scratch/jmoylan/AMS-530-Principles-of-Parallel-Computing/project1/analysis/analysis-ordered.py

