from mpi4py import MPI
import numpy as np
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument("--matrix_size", type=str, required=True)
args = parser.parse_args()

start_time = time.perf_counter()
N = int(args.matrix_size)

COMM = MPI.COMM_WORLD
size = COMM.Get_size()
rank = COMM.Get_rank()

grid_rows = int(np.sqrt(size))

grid_cols = grid_rows

# Get coordinates of each element
y = rank // grid_cols
x = rank % grid_cols

# Create row communicators for each row of the matrix
row_comm = COMM.Split(color=y, key=x)

local_A = None
local_B = None

if rank == 0:
    rng = np.random.default_rng(42)
    A = rng.uniform(-1, 1, size=(N, N))
    B = rng.uniform(-1, 1, size=(N, N))
    
    row_splits_A = np.vsplit(A, grid_rows)
    grid_A = [np.hsplit(row_block, grid_cols) for row_block in row_splits_A]

    row_splits_B = np.vsplit(B, grid_rows)
    grid_B = [np.hsplit(row_block, grid_cols) for row_block in row_splits_B]

    # give each process its corresponding block of A and B
    for r in range(grid_rows):
        for c in range(grid_cols):
            target_rank = r * grid_cols + c
            data = {
                'A': grid_A[r][c],
                'B': grid_B[r][c]
            }

            if target_rank == 0:
                local_A = grid_A[r][c]
                local_B = grid_B[r][c]
            else:
                COMM.send(data, dest=target_rank)
else:
    data = COMM.recv(source=0)
    local_A = data['A']
    local_B = data['B']


def main(local_A, local_B):
    # Uninterrupted memory access for local arrays for SendRecv to prevent memory issues
    local_A = np.ascontiguousarray(local_A)
    local_B = np.ascontiguousarray(local_B)
    
    local_C = np.zeros((local_A.shape[0], local_B.shape[1]))

    # Initial alignment of A and B blocks
    left_rank = y * grid_cols + (x - 1 + grid_cols) % grid_cols
    right_rank = y * grid_cols + (x + 1) % grid_cols
    up_rank = ((y - 1 + grid_rows) % grid_rows) * grid_cols + x
    down_rank = ((y + 1) % grid_rows) * grid_cols + x

    for _ in range(y):
        rec_A = np.empty_like(local_A)
        COMM.Sendrecv(local_A, dest=left_rank, recvbuf=rec_A, source=right_rank)
        local_A = np.ascontiguousarray(rec_A)

    for _ in range(x):
        rec_B = np.empty_like(local_B)
        COMM.Sendrecv(local_B, dest=up_rank, recvbuf=rec_B, source=down_rank)
        local_B = np.ascontiguousarray(rec_B)

    # Cannon's algorithm main loop
    for step in range(grid_cols):
        local_C += np.dot(local_A, local_B)

        if step < grid_rows - 1:
            recv_A = np.empty_like(local_A)
            recv_B = np.empty_like(local_B)
            
            COMM.Sendrecv(local_A, dest=left_rank, recvbuf=recv_A, source=right_rank)
            COMM.Sendrecv(local_B, dest=up_rank, recvbuf=recv_B, source=down_rank)
            
            local_A = np.ascontiguousarray(recv_A)
            local_B = np.ascontiguousarray(recv_B)

    gathered = COMM.gather(local_C, root=0)
    
    if rank == 0:
        full_C = np.block([[gathered[r * grid_cols + c] for c in range(grid_cols)] for r in range(grid_rows)])
        end_time = time.perf_counter()
        print(f"Total execution time: {end_time - start_time} seconds\n")
        t0 = time.perf_counter(); _ = A @ B; print(f"{time.perf_counter() - t0:.6f} seconds for serial baseline")
        print("Does using Cannon's algorithm result match the expected result?")
        print(np.allclose(full_C, A@B))
        
if __name__ == "__main__":
    main(local_A, local_B)
