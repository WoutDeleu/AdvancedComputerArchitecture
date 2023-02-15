import math
import time
import numpy as np
from numba import cuda

N = 1000000
input_arr = np.arange(N)
output_arr = np.zeros_like(input_arr)

input_cpu = input_arr
output_cpu = output_arr


# Define kernel (GPU) function ...
@cuda.jit
def flip(input_arr, output_arr):
    i = cuda.threadIdx.x + 1024 * (cuda.blockIdx.x - 1)
    output_arr[i] = input_arr[N - i - 1]


# Call kernel function ...
# Time it
numberBlocks = math.ceil(N / 1024)
numberThreads = N / numberBlocks;
flip[numberBlocks, numberBlocks](input_arr, output_arr)
start = time.time()
# 1 block, N threads
# code uitgevoerd voor N/2 threads
flip[numberBlocks, numberBlocks](input_arr, output_arr)
cuda.synchronize()
total = time.time() - start
print("GPU RESULTS:")
print(total)
# print(input_arr)
# print(output_arr)
print()

start = time.time()
for i in range(0, int(N / 2)):
    temp = input_cpu[i]
    output_cpu[i] = input_cpu[N - i - 1]
    output_cpu[N - 1 - i] = temp
total = time.time() - start
print("CPU RESULTS:")
print(total)
# print(input_cpu)
# print(output_cpu)
