import math
import sys
import time
from timeit import timeit
import pandas as pd
import matplotlib.pyplot as plt


import numpy as np
from numba import cuda

# report: evaluation complexity changes with input size, complexity curves
def matrix_multiplication_CPU(A, B, C):  # Result
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i][j] += A[i][k] * B[k][j]


# report: evaluation complexity changes with input size, complexity curves
@cuda.jit
def matrix_multiplication_GPU(A, B, resulting_matrix):  # Result
    # x = row, y = column
    x, y = cuda.grid(2)
    for i in range(x, A.shape[0], cuda.blockDim.y * cuda.gridDim.y):
        for j in range(y, A.shape[1], cuda.blockDim.x * cuda.gridDim.x):
            # if i < A.shape[0] and j < B.shape[1]:
            sum = 0
            for k in range(A.shape[1]):
                sum += A[i, k] * B[k, j]
            resulting_matrix[i, j] = sum


@cuda.jit
def matrix_multiplication_GPU_shared_memory(A, B, resulting_matrix):
    A_shared = cuda.shared.array(shape=block_size, dtype=np.int32)
    B_shared = cuda.shared.array(shape=block_size, dtype=np.int32)
    i, j = cuda.grid(2)

    # tx = kolom
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    amount_of_blocks = cuda.gridDim.x  # blocks per grid

    sum = 0
    for x in range(amount_of_blocks):
        if j < A.shape[0] and (tx + x * block_size[0]) < A.shape[1]:
            A_shared[ty, tx] = A[j, tx + x * block_size[0]]
        if i < B.shape[1] and (ty + x * block_size[0]) < B.shape[0]:
            B_shared[ty, tx] = B[ty + x * block_size[0], i]

        cuda.syncthreads()

        for y in range(block_size[0]):
            sum += A_shared[ty, y] * B_shared[y, tx]

        cuda.syncthreads()

    if j < resulting_matrix.shape[0] and i < resulting_matrix.shape[1]:
        resulting_matrix[j, i] = sum


size = 1

timing_cpu = {}
timing_cpu_np = {}
timing_gpu_naive = {} 
timing_gpu_shared = {}


print("size: ", format(size))
block_size = 32, 32
# 1024 threads per block
number_blocks_x = math.ceil(size / block_size[0])
number_blocks_y = math.ceil(size / block_size[1])

# can also be int
amount_of_blocks = number_blocks_x, number_blocks_y


# precompiling
A = np.random.randint(10, size=(size, size))
B = np.random.randint(10, size=(size, size))
resulting_matrix = np.zeros((size, size))

matrix_multiplication_GPU[amount_of_blocks, block_size](A, B, resulting_matrix)
matrix_multiplication_GPU_shared_memory[amount_of_blocks, block_size](A, B, resulting_matrix)


while size < 1000:

    A = np.random.randint(10, size=(size, size))
    B = np.random.randint(10, size=(size, size))
    resulting_matrix = np.zeros((size, size))
    a_dev = cuda.to_device(A)
    b_dev = cuda.to_device(B)
    res_dev = cuda.to_device(resulting_matrix)

    cpu_np = lambda: np.matmul(A, B)
    timing_cpu_np[size] = timeit(cpu_np, number=100)

    if(size < -1):
        cpu = lambda: matrix_multiplication_CPU(A, B, resulting_matrix)
        timing_cpu[size] = timeit(cpu, number=100)
        resulting_matrix = np.zeros((size, size))

    # gpu_naive = lambda: matrix_multiplication_GPU[amount_of_blocks, block_size](A, B, resulting_matrix)
    gpu_naive = lambda: matrix_multiplication_GPU[amount_of_blocks, block_size](a_dev, b_dev, res_dev)
    timing_gpu_naive[size] = timeit(gpu_naive, number=100)
    start = time.time()
    for i in range(20):
        gpu_naive()
        cuda.synchronize()
    timing_gpu_naive[size] = time.time() - start
    resulting_matrix = np.zeros((size, size))
    
    # print("GPU")
    # print(resulting_matrix)

    # gpu_shared = lambda: matrix_multiplication_GPU_shared_memory[amount_of_blocks, block_size](A, B, resulting_matrix)
    gpu_shared = lambda: matrix_multiplication_GPU_shared_memory[amount_of_blocks, block_size](a_dev, b_dev, res_dev)
    # timing_gpu_shared[size] = timeit(gpu_shared, number=100)
    start = time.time()
    for i in range(20):
        gpu_shared()
        cuda.synchronize()
    timing_gpu_shared[size] = time.time() - start

        # print("Shared Memory")
    # print(resulting_matrix)
    # size *= 2
    size *= 4

print(timing_cpu)
print(timing_cpu_np)
print(timing_gpu_shared)
print(timing_gpu_naive)

# x-as: size van de dictionaries
sizes = list(timing_gpu_shared.keys())

# y-as: tijden van de drie dictionaries
cpu_times = list(timing_cpu.values())
cpu_np_times = list(timing_cpu_np.values())
gpu_naive_times = list(timing_gpu_naive.values())
gpu_shared_times = list(timing_gpu_shared.values())

# Plot de drie lijnen op dezelfde grafiek
#plt.plot(sizes, cpu_times, label='CPU')
# plt.plot(sizes, cpu_np_times, label='CPU Numpy')
plt.plot(sizes, gpu_naive_times, label='GPU Naive')
plt.plot(sizes, gpu_shared_times, label='GPU Shared Memory')

# Voeg de legenda toe
plt.legend()

# Voeg titel en labels toe
plt.title('Times for different implementations')
plt.xlabel('Size')
plt.ylabel('Time (ms)')
plt.yscale('log')

# Toon de grafiek
plt.show()