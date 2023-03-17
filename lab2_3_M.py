import math
import time

import numpy as np
from numba import cuda

arraySize = 3

# 1024 threads per block
number_blocks_x = math.ceil(arraySize / 32)
number_blocks_y = math.ceil(arraySize / 32)

# amount of threads, can also be int
block_size = 32, 32

# can also be int
amount_of_blocks = number_blocks_x, number_blocks_y


# summed is a one dimensional array
@cuda.jit
def sum_axis_matrix_atomic(matrix, summed, axis):
    # x, y coord of thread within grid, x row y column
    x, y = cuda.grid(2)
    #x = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    #y = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y


    #undefined behaviour possible if we dont add this statement
    if x < input_matrix.shape[0] and y < input_matrix.shape[1]:
        if axis == 0:
            #y is column!
            cuda.atomic.add(summed, y, matrix[x][y])
        if axis == 1:
            cuda.atomic.add(summed, x, matrix[x][y])
        # todo: make dynamic


#input always 2^N, INTERLEAVED REDUCTION!
@cuda.jit
def sum_axis_matrix_reduction(matrix, summed):

    x, y = cuda.grid(2)

    #i = iterator over array
    for i in range(1, cuda.blockDim.x):
        if x % (2 * i) == 0:
            matrix[x][y] += matrix[x + i][y]
        i *= 2
    cuda.syncthreads()
    if y == 0:
        summed[y] = matrix[0][y]


summed_array_atomic = np.zeros(arraySize)
input_matrix = np.random.randint(10, size=(arraySize, arraySize))

axis = 0
print("Input:")
print(input_matrix)
print()

start = time.time()
sum_axis_matrix_atomic[amount_of_blocks, block_size](input_matrix, summed_array_atomic, axis)
total = time.time() - start
print("GPU RESULTS ATOMIC:")
print(summed_array_atomic)
print(total)

print()

start = time.time()
summed = input_matrix.sum(axis=0)
total = time.time() - start
print("CPU RESULTS ATOMIC")
print(summed)
print(total)
print()

summed_array_red = np.zeros(arraySize)
start = time.time()
sum_axis_matrix_reduction[amount_of_blocks, block_size](input_matrix, summed_array_red)
total = time.time() - start
print(summed_array_red)


