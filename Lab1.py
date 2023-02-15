import math
import time
import numpy as np
from numba import cuda

N = 4000
input_arr = np.arange(N)
output_arr = np.zeros_like(input_arr)

input_cpu = input_arr
output_cpu = output_arr


# Define kernel (GPU) function ...
@cuda.jit
def flip(input_arr, output_arr):
    i = cuda.threadIdx.x + 1024 * (cuda.blockIdx.x -1)
    temp = input_arr[i]
    output_arr[i] = input_arr[N - i - 1]
    output_arr[N - 1 - i] = temp

# Call kernel function ...
# Time it
numberBlocks = math.ceil((N/2) / 1024)
flip[numberBlocks, int(N/2)](input_arr, output_arr)
start = time.time()
#1 block, N/2 threads
#code uitgevoerd voor N/2 threads
flip[numberBlocks, int(N/2)](input_arr, output_arr)
cuda.synchronize()
total = time.time() - start
print("GPU RESULTS:")
print(total)
#print(input_arr)
#print(output_arr)
print()

start = time.time()
for i in range(0, int(N / 2)):
    temp = input_cpu[i]
    output_cpu[i] = input_cpu[N - i - 1]
    output_cpu[N - 1 - i] = temp
total = time.time() - start
print("CPU RESULTS:")
print(total)
#print(input_cpu)
#print(output_cpu)
