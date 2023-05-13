import math

import numpy as np
from numba import cuda, float32

size = 10

A = np.random.randint(10, size=(size, size))
B = np.random.randint(10, size=(size, size))
resulting_matrix = np.zeros((size, size))

print("size: ", format(size))

threads_per_block = 32
block_size = 32, 32
# 1024 threads per block
number_blocks_x = math.ceil(size / block_size[0])
number_blocks_y = math.ceil(size / block_size[1])

# can also be int
amount_of_blocks = number_blocks_x, number_blocks_y


@cuda.jit
def matrix_multiplication_GPU_shared_memory(A, B, resulting_matrix):
    A_shared = cuda.shared.array(shape=block_size, dtype=np.int32)
    B_shared = cuda.shared.array(shape=block_size, dtype=np.int32)

    # relatieve positie
    thread_x = cuda.threadIdx.x
    thread_y = cuda.threadIdx.y
    amount_of_blocks = cuda.gridDim.x

    # absolute positie
    x = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    y = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    sum = 0
    for i in range(amount_of_blocks):
        if y < A.shape[0] and (thread_x + i * block_size[0]) < A.shape[1]:
            A_shared[thread_y, thread_x] = A[y, thread_x + i * block_size[0]]
        if x < B.shape[1] and (thread_y + i * block_size[0]) < B.shape[0]:
            B_shared[thread_y, thread_x] = B[thread_y + i * block_size[0], x]

        cuda.syncthreads()

        for j in range(block_size[0]):
            sum += A_shared[thread_y, j] * B_shared[j, thread_x]

        cuda.syncthreads()

    if y < resulting_matrix.shape[0] and x < resulting_matrix.shape[1]:
        resulting_matrix[y, x] = sum


@cuda.jit
def matrix_multiplication_GPU(A, B, resulting_matrix):

    # x = column, y = row
    x = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    y = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    for i in range(x, A.shape[1], cuda.blockDim.x * cuda.gridDim.x):
        for j in range(y, A.shape[0], cuda.blockDim.y * cuda.gridDim.y):
            sum = 0
            for k in range(A.shape[1]):
                sum += A[j, k] * B[k, i]
            resulting_matrix[j, i] = sum



