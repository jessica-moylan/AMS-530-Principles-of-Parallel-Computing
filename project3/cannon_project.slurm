#!/usr/bin/env bash
#SBATCH --job-name=matrix_cannon_algo
#SBATCH --partition=short-96core-shared
#SBATCH --output=benchmark_results_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=10:00

module load mpi4py/latest

# Define the core counts and matrix sizes to test
CORES_LIST=(4 16 64)
MATRIX_SIZES=(256 1024 4096)

echo "Starting Matrix Multiplication Benchmarks..."
echo "============================================="

for P in "${CORES_LIST[@]}"; do
    for N in "${MATRIX_SIZES[@]}"; do
        echo "---------------------------------------------"
        echo "Running with P = $P cores, N = $N matrix size"
        echo "---------------------------------------------"
        
        # Run using mpirun with the specific core count P
        mpirun -np "$P" python /gpfs/scratch/jmoylan/AMS-530-Principles-of-Parallel-Computing/project3/cannon_algorithm.py --matrix_size "$N"
        
        echo ""
    done
done

echo "All benchmarks completed!"
