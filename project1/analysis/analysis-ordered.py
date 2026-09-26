from mpi4py import MPI

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank() 
size = COMM.Get_size()

# Makes sure that all processors reach this point before starting the timer
COMM.Barrier()
start_time = MPI.Wtime()

data = None 
if rank == 0:
    print(f"Hello from Processor {rank}")

    COMM.send(data, dest=rank + 1)  
else:
    COMM.recv(source=rank-1)
    print(f"Hello from Processor {rank}")

    if rank < size - 1:
        COMM.send(data, dest=rank + 1)

# Makes sure that all processors reach this point before stopping the timer
COMM.Barrier()
end_time = MPI.Wtime()

total_time = end_time - start_time

if rank == 0:
    print(f"\n==========================================")
    print(f"Total Execution Time: {total_time:.6f} seconds")
    print(f"==========================================")
    print()
    print()
