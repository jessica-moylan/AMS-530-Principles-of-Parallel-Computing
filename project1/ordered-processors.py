#!/usr/bin/env python
# This has to be done with MPI since we need to use 16 processors
# And we must use the SeaWulf cluster which is why we can not use multiprocessing

# multiprocessing was designed for single computer, with multiple cores

from mpi4py import MPI

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank() # gets the processor ID
size = COMM.Get_size() # says how many processors are being used (defined when starting the program) with slurm

data = None # we are not trasmitting data, so this is just a placeholder
if rank == 0:
    print(f"Hello from Processor {rank}")
    # Send a message to other processos that 0 has arrived

    # COMM.send(data, dest, tag=0 )
    # data is what we are sending, not important for this problem
    # dest is the rank of the processor we are sending the data to
    COMM.send(data, dest=rank + 1)  # Send a message to the next processor in order
else:
    COMM.recv(source=rank-1)
    print(f"Hello from Processor {rank}")

    # Forward signal to the next processor IF this isn't the last rank
    if rank < size - 1:
        COMM.send(data, dest=rank + 1)
