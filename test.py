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





res = np.matmul(A, B)
print(res)


normal = matrix_multiplication_GPU[amount_of_blocks, block_size](A, B, resulting_matrix)
print(resulting_matrix)

resulting_matrix = np.zeros((size, size))



shared = matrix_multiplication_GPU_shared_memory[amount_of_blocks, block_size](A, B, resulting_matrix)
print(resulting_matrix)