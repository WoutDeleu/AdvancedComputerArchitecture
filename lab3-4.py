import math
import sys
from timeit import timeit
import pandas as pd


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
    sA = cuda.shared.array(shape=block_size, dtype=np.int32)
    sB = cuda.shared.array(shape=block_size, dtype=np.int32)
    i, j = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    amount_of_blocks = cuda.gridDim.x  # blocks per grid
    #
    # if x >= resulting_matrix.shape[0] and y >= resulting_matrix.shape[1]:
    #     # Quit if (x, y) is outside of valid C boundary
    #     return

    sum = 0
    for x in range(amount_of_blocks):
        sA[tx, ty] = A[i, ty + x * block_size[0]]
        sB[tx, ty] = B[tx + x * block_size[1], j]

        cuda.syncthreads()

        for y in range(block_size[0]):
            sum += sA[tx, y] * sB[y, ty]

        cuda.syncthreads()

    resulting_matrix[i, j] = sum


size = 1

timing_cpu = pd.Series(dtype=float)
timing_gpu_naive = pd.Series(dtype=float)
timing_gpu_shm = pd.Series(dtype=float)

while size < 100000:
    print("size: {}".format(size))
    block_size = 32, 32
    # 1024 threads per block
    number_blocks_x = math.ceil(size / block_size[0])
    number_blocks_y = math.ceil(size / block_size[1])

    # can also be int
    amount_of_blocks = number_blocks_x, number_blocks_y

    A = np.random.randint(10, size=(size, size))
    B = np.random.randint(10, size=(size, size))
    resulting_matrix = np.zeros((size, size))

    # print(A)
    # print(B)

    cpu = lambda: matrix_multiplication_CPU(A, B, resulting_matrix)
    pd.concat([timing_cpu, pd.Series(timeit(cpu, number=1000))])
    # print("CPU")
    # print(resulting_matrix)

    gpu_naive = lambda: matrix_multiplication_GPU[amount_of_blocks, block_size](A, B, resulting_matrix)
    pd.concat([timing_gpu_naive, pd.Series(timeit(gpu_naive, number=1000))])   # print("GPU")
    # print(resulting_matrix)
    # sys.stdout.flush()

    gpu_shm = lambda: matrix_multiplication_GPU_shared_memory[amount_of_blocks, block_size](A, B, resulting_matrix)
    pd.concat([timing_gpu_shm, pd.Series(timeit(gpu_shm, number=1000))])
    # print("Shared Memory")
    # print(resulting_matrix)
    # sys.stdout.flush()
    size *= 2
    if (size > 100):
        break

print(timing_cpu)
print(timing_gpu_shm)
print(timing_gpu_naive)