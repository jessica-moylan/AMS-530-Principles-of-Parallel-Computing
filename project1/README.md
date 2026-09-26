# Project 1 Question 2:

## Project Description

Goal: With parallel computing the sequence to which processors deliver messages are non-deterministic.
There are cases however, where messages sent must be deterministic. So the project is trying to determine a way
to show this to ensure consistant output so that the output will always be ...

```text
Hello from Processor 0
Hello from Processor 1
Hello from Processor 2
...
Hello from Processor 9
```

## Program Structure

Files included on this project
- [ordered-processors.py](project1/ordered-processors.py): This is the source code where the code logic is
- [project1output.txt](project1/project1output.txt): An example of the ordered output from the code
- [project1.slurm](project1/project1.slurm): The bash script submitted to Seawulf (with `sbatch project1.slurm`) that started the program
   - ( Note to grader, to fully replicate this code, the path for where file is located must reflect your own directory change should be `/gpfs/scratch/<SBUID>/project1/ordered-processors.py`)
- [/analysis](project1/analysis): the folder that was used to hold the files used to test the timing between using a different number of processors, the slurm file was ran multiple times, changing the number of processors between 8, 16, 32, and 40 

The main program strcture is the following in puesdocode:

```text
IF rank IS EQUAL TO 0 THEN
    PRINT "Hello from Processor 0"
    SEND data TO Processor (rank + 1)
ELSE
    RECEIVE data FROM Processor (rank - 1)
    PRINT "Hello from Processor " + rank
    
    IF not at the last processor THEN
        SEND data TO Processor (rank + 1)
```

What this is demonstrating is that unless the 0th processor is seen, all other processors are blocked from
continuing at the MPI.COMM_WORLD.recv call. 

Once the 0th processor sends its message and prints `Hello from Processor 0`,
the process waits until the 1st processor recieves the data, then prints its hello. The if statement inside the else
insures that the message will continue to be sent in order to the next processor until the last. In where the program
completes.


## Results

Using COMM.send and COMM.recv I was able to create a program that determinstically returns the processors in order.
More information on the exact output can be found at [project1output.txt](project1/project1output.txt).

For examples of outputs with a different number of processors you can check out [timeanlysis32.txt](project1/analysis/timeanlysis32.txt) or any of the other `.txt` files in the `analysis` folder. 

## Analysis
In addition to the main code, I looked to see how, when using different number of processors, how the time to complete the process would change. As expected, the time to finish, was more or less linear to the amount of processors used. This increase in time between the nuber of processors is due to the increased latency that is created due to the communication between processors

![Graph](project1/analysis/processor_time_plot.png)

Due to the nature of parallel computing, processors will be activated and complete their processing at different times, in part to the latency due to the communication between the nodes. 

Another thing to note, forcing processors to behave sequentially does increase run time by a decent amount, a test done [here](project1/analysis/sequential.py) shows that for printing `hello` 40 times in a for loop takes 0.000242 seconds but doing this with the 40 processors took 0.003278 seconds, an increase of 0.003036 seconds. This implies that it is best to minimize the amount of time that the processors should work in sync, and as much work should be done independently as possible.


## Additional Notes

The `order-processors.py` file contains notes and comments about the code while learning about MPI 
