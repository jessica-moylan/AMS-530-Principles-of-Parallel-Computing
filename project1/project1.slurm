#!/usr/bin/env bash

#SBATCH --job-name=project1
#SBATCH --partition=short-40core-shared
#SBATCH --output=project1output.txt
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --time=05:00

module load mpi4py/latest

mpirun -np 16 python /gpfs/scratch/jmoylan/project1/ordered-processors.py
