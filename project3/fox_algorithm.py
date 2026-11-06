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

    # Get neighbors for vertical shifting
    up_rank = ((y - 1 + grid_rows) % grid_rows) * grid_cols + x
    down_rank = ((y + 1) % grid_rows) * grid_cols + x

    for step in range(grid_rows):
        root_col = (y + step) % grid_cols

        # Broadcast A across the row
        if x == root_col:
            broadcast_A = np.ascontiguousarray(local_A)
        else:
            broadcast_A = np.empty_like(local_A)

        # row_comm.Bcast broadcasts from root_col (which has rank root_col in row_comm)
        row_comm.Bcast(broadcast_A, root=root_col)

        local_C += np.dot(broadcast_A, local_B)

        # Shift B blocks upwards along the column using COMM.Sendrecv with a pre-allocated recvbuf
        recv_B = np.empty_like(local_B)
        COMM.Sendrecv(local_B, dest=up_rank, recvbuf=recv_B, source=down_rank)
        
        # Update local_B and ensure contiguity for the next iteration
        local_B = np.ascontiguousarray(recv_B)

    gathered = COMM.gather(local_C, root=0)
    
    if rank == 0:
        full_C = np.block([[gathered[r * grid_cols + c] for c in range(grid_cols)] for r in range(grid_rows)])
        end_time = time.perf_counter()
        print(f"Total execution time: {end_time - start_time} seconds\n")
        print()
        t0 = time.perf_counter(); _ = A @ B; print(f"{time.perf_counter() - t0:.6f} seconds for serial baseline")
        print("Does using Fox's algorithm result match the expected result?")
        print(np.allclose(full_C, A@B))
        
if __name__ == "__main__":
    main(local_A, local_B)
