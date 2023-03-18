import math
import time

import numpy as np
from numba import cuda

arraySize = 100
# amount of threads, can also be in
if arraySize < 32:
    block_size = arraySize, arraySize
else:
    block_size = 32, 32

# 1024 threads per block
number_blocks_x = math.ceil(arraySize / 32)
number_blocks_y = math.ceil(arraySize / 32)

# amount of threads, can also be int
block_size = 32, 32

# can also be int
amount_of_blocks = number_blocks_y, number_blocks_x


# summed is a one dimensional array
@cuda.jit
def sum_axis_matrix_atomic(matrix, summed, axis):
    # x, y coord of thread within grid, x row and y column

    x = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    y = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    # undefined behaviour possible if we don't add this statement
    # shape[0] is amount of rows
    if x < matrix.shape[1] and y < matrix.shape[0]:
        if axis == 0:
            cuda.atomic.add(summed, x, matrix[y][x])
        if axis == 1:
            cuda.atomic.add(summed, y, matrix[y][x])

# reduction is way faster!!! No atomic operations!!
# input always 2^N, INTERLEAVED REDUCTION!
@cuda.jit
def sum_axis_matrix_reduction(matrix, summed):
    thread_id = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x + cuda.blockDim.y * cuda.blockIdx.y
    #thread_id = (cuda.blockId.x * (cuda.blockDim.x * cuda.blockDim.y)) + (cuda.threadIdx.y * cuda.blockDim.x) + cuda.threadIdx.x

    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    s = 1
    while s < cuda.blockDim.x:
        index = 2 * s * thread_id
        # if i < cuda.blockDim.x and y < cuda.blockDim.y:
        if index + s < matrix.shape[0] and y < matrix.shape[1]:
            matrix[index][y] += matrix[index + s][y]
        s = s * 2
        cuda.syncthreads()
    if index == 0 and y < cuda.blockDim.y:
        summed[y] = matrix[index][y]


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
summed = input_matrix.sum(axis=axis)
total = time.time() - start
print("CPU RESULTS ATOMIC")
print(summed)
print(total)
print()

summed_array_red = np.zeros(arraySize)
start = time.time()
sum_axis_matrix_reduction[amount_of_blocks, block_size](input_matrix, summed_array_red)
total = time.time() - start
print("GPU RESULTS REDUCTION")
print(summed_array_red)


