import time

start_time = time.time()

for i in range(40):
    print(f"Hello from step {i}")

end_time = time.time()
total_time = end_time - start_time
print(f"Total Execution Time: {total_time:.6f} seconds")