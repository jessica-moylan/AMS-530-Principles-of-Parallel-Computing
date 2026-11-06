# Project 2 Question 4:

## Project Description

Goal: How do you implement different parallel computing algorithms for matrix multiplication?

This project is looking at two popular methods, Cannon's and Fox's algorithm for two `N`x`N` matrixes that have values between (-1,1]

## Program Structure

Files included on this project
- [fox_matrix.py](/project3/fox_matrix.py): 

- [fox_output.txt](/project3/fox_output.txt):

- [cannon_matrix.py](/project3/cannon_matrix.py): 

- [cannon_output.txt](/project3/cannon_output.txt): 

- [fox_project3.slurm](/project3/fox_project3.slurm): The bash script submitted to Seawulf (with `sbatch fox_project3.slurm`) that started the program.

- [cannon_project3.slurm](/project3/cannon_project3.slurm): The bash script submitted to Seawulf (with `sbatch cannon_project3.slurm`) that started the program.

## Code Analysis

### Algorithm Description:

#### Fox Algorithm
The General idea of the algorithm is that within each row, a single element is broadcasted to all other processors within the same row of matrix A. We start at the element (i,i) than each new iteration j, we broadcast (i,i+j)

After the broadcast, we use a local subsect of the resulting matrix and populate that with the result of the local matrix multiplication. Each iteration after the local result matrices are added together

We also need to circulaly rotate the local data for the B matrix. Each coloumn is roatated to the processor at index one above them. 

We than keep repeating this process until we have done it until each element of a row has been broadcasted, which ends of being sqrt(number of processors) times


The Puesdocode:
``` text
q = sqrt(number of processors)

# Each processor P(i, j) has local matrices A_local, B_local of size (n/q) x (n/q)
# Assign each processor in order to keep the original layout of the matrixes

C_local = 0.0  # Initialize local result sub-matrix to zero in the shape of

# Determine neighbor ranks for vertical communication in the grid
up_rank   = ((y - 1 + grid_rows) mod grid_rows) * grid_cols + x
down_rank = ((y + 1) mod grid_rows) * grid_cols + x

For step = 0 to grid_rows - 1:
    # Determine the root column in the current row for broadcasting A
    root_col = (y + step) mod grid_cols

    # Prepare buffer for broadcasting A across the processor row
    If x == root_col then
        broadcast_A = local_A
    Else
        broadcast_A = empty buffer with same shape as local_A
    End If

    # Broadcast A from the root column to all processors in the same row
    Broadcast broadcast_A from processor column root_col across row_comm

    # Multiply received A sub-matrix with local B and accumulate into local C
    local_C = local_C + (broadcast_A * local_B)

    # Shift local_B blocks upward along the processor column
    recv_B = empty buffer with same shape as local_B
    Send local_B to up_rank and receive recv_B from down_rank simultaneously
    
    # Update local_B with the shifted data
    local_B = recv_B
End For
```
#### Cannon Algorithm

## Results


## Analysis

### Perforance Differences

### Perforance Trends

### Scaliability