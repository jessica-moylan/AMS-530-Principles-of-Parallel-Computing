#!/usr/bin/env bash

#SBATCH --job-name=inorder-in-c
#SBATCH --output=ordered.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --time=05:00
#SBATCH --partition=short-40core-shared

# load an MPI module
module load mvapich2/gcc12.1/2.3.7

# set env variables which may help performance

export MV2_HOMOGENEOUS_CLUSTER=1
export MV2_ENABLE_AFFINITY=0

# compile the code with the mpi compiler wrapper
mpicxx /gpfs/home/jmoylan/AMS-530-Principles-of-Parallel-Computing/project1/additional-practice/inorder-process.cpp -o inorder-process

# execute the code with MPI
mpirun ./inorder-process
