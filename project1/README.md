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
- ordered-processors.py : This is the main project code where the code logic is
- project1output.txt : An example of the ordered output from the code
- project1.slurm : The bash script submitted to Seawulf that started the program


The main program strcture is the following in puesdocode:
```text
IF rank IS EQUAL TO 0 THEN
    PRINT "Hello from Processor 0"
    SEND data TO Processor (rank + 1)
ELSE
    RECEIVE data FROM Processor (rank - 1)
    PRINT "Hello from Processor " + rank
    
    IF rank IS LESS THAN (size - 1) THEN
        SEND data TO Processor (rank + 1)
    END IF
END IF
```
What this is demonstrating is that unless the 0th processor is seen, all other processors are blocked from
continuing at the MPI.COMM_WORLD.recv call. Once the 0th processor sends its message and prints `Hello from Processor 0`,
the process waits until the 1st processor recieves the data, then prints its hello. The if statement inside the else
insures that the message will continue to be sent to the next processor until the last. In where the program
completes.


## Results

Using COMM.send and COMM.recv I was able to create a program that determinstically returns the processors in order.
More information on the exact output can be found in `project1output.txt`

## Analysis

The program is quite efficent as it proceeds as soon as the processor before it recieves its data resulting 
in a deterministic output.


## Additional Notes

The `order-processors.py` file contains notes and comments about the code while learning about MPI 
