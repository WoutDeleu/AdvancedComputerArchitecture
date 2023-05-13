import math

import numpy as np
from numba import cuda, float32

size = 1000

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




@cuda.jit
def shared(A, B, resulting_matrix):
    tx = cuda.threadIdx.x
    x = tx + cuda.blockDim.x * cuda.blockIdx.x

    ty = cuda.threadIdx.y
    y = ty + cuda.blockDim.y * cuda.blockIdx.y

    A_shared = cuda.shared.array(shape=block_size, dtype=np.int32)
    B_shared = cuda.shared.array(shape=block_size, dtype=np.int32)

    sum = 0

    # copy naar shared memory
    for i in range(cuda.gridDim.x):

        # nodig om unexpected behaviour te vermijden!
        A_shared[ty, tx] = 0
        B_shared[ty, tx] = 0

        if y < A.shape[0] and tx + i * block_size[1] < A.shape[1]:
            A_shared[ty, tx] = A[y, tx + i * block_size[1]]

        if x < A.shape[1] and ty + i * block_size[0] < A.shape[0]:
            B_shared[ty, tx] = B[ty + i * block_size[0], x]

        # na iteratie syncen!
        cuda.syncthreads()

        for j in range(block_size[1]):
            sum += A_shared[ty, j] * B_shared[j, tx]

        cuda.syncthreads()

    if x < resulting_matrix.shape[1] and y < resulting_matrix.shape[0]:
        resulting_matrix[y, x] = sum


def matrix_multiplication_CPU(A, B, resulting_matrix):
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                resulting_matrix[i][j] += A[i][k] * B[k][j]


res = np.matmul(A, B)
print(res)


# normal = matrix_multiplication_GPU[amount_of_blocks, block_size](A, B, resulting_matrix)
# print(resulting_matrix)
#
# resulting_matrix = np.zeros((size, size))

# mult = matrix_multiplication_CPU(A, B, resulting_matrix)
# print("normal", resulting_matrix)
# resulting_matrix = np.zeros((size, size))

shared = shared[amount_of_blocks, block_size](A, B, resulting_matrix)
print("shared", resulting_matrix)
resulting_matrix = np.zeros((size, size))

#
# f = fast_matmul[amount_of_blocks, block_size](A, B, resulting_matrix)
# print("fast", resulting_matrix)