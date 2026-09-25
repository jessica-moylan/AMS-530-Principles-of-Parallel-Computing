#include <iostream>
#include <mpi.h>

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int token = 0; // Placeholder data to send 

    if (rank == 0) {
        std::cout << "Hello from Processor " << rank << std::endl;

        MPI_Send(&token, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);

    } else {
        // Wait to receive message from the previous processor (rank - 1)
        MPI_Recv(&token, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::cout << "Hello from Processor " << rank << std::endl;

        // Forward token to the next processor IF this isn't the last process
        if (rank < size - 1) {
            MPI_Send(&token, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
    }
 
    MPI_Finalize();
    return 0;
}
