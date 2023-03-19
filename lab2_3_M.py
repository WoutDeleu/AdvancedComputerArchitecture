import math
import time

import numpy as np
from numba import cuda

arraySize = 10000
block_size = 32, 32

# 1024 threads per block
number_blocks_x = math.ceil(arraySize / block_size[1])
number_blocks_y = math.ceil(arraySize / block_size[0])

# can also be int
amount_of_blocks = number_blocks_y, number_blocks_x


# summed is a one dimensional array
@cuda.jit
def sum_axis_matrix_atomic(matrix, summed, axis):
    x = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    y = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    if axis == 0:
        for i in range(x, matrix.shape[1], cuda.blockDim.x * cuda.gridDim.x):
            for j in range(y, matrix.shape[0], cuda.blockDim.y * cuda.gridDim.y):
                cuda.atomic.add(summed, i, matrix[j][i])
    else:
        for j in range(y, matrix.shape[0], cuda.blockDim.y * cuda.gridDim.y):
            for i in range(x, matrix.shape[1], cuda.blockDim.x * cuda.gridDim.x):
                cuda.atomic.add(summed, j, matrix[j][i])



# reduction is way faster!!! No atomic operations!!
# input always 2^N, INTERLEAVED REDUCTION!
@cuda.jit
def sum_axis_matrix_reduction(matrix, summed):
    thread_id = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x + cuda.blockDim.y * cuda.blockIdx.y

    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    s = 1
    while s < cuda.blockDim.x:
        index = 2 * s * thread_id
        # if i < cuda.blockDim.x and y < cuda.blockDim.y:
        if index + s < matrix.shape[0] and y < matrix.shape[1]:
            matrix[index][y] += matrix[index + s][y]
        cuda.syncthreads()
        s = s * 2
    if index == 0 and y < cuda.blockDim.y:
        summed[y] = matrix[index][y]


summed_array_atomic = np.zeros(arraySize)
input_matrix = np.random.randint(10, size=(arraySize, arraySize))
# input_matrix = np.ones((arraySize, arraySize))

# 0 is column
axis = 0
print("Input:")
print(input_matrix)
print()

input_matrix_2 = input_matrix.copy()
input_matrix_3 = input_matrix.copy()

sum_axis_matrix_atomic[amount_of_blocks, block_size](input_matrix, summed_array_atomic, axis)

times = []

for i in range(10):
    summed_array_atomic = np.zeros(arraySize)
    start = time.time()
    sum_axis_matrix_atomic[amount_of_blocks, block_size](input_matrix, summed_array_atomic, axis)
    total = time.time() - start
    times.append(total)
print("GPU RESULTS ATOMIC:")
print(summed_array_atomic)
print(times)
print(np.average(times))




print()

start = time.time()
summed = input_matrix.sum(axis=axis)
total = time.time() - start
print("CPU RESULTS")
print(summed)
print(total)
print()






summed_array_red = np.zeros(arraySize)
sum_axis_matrix_reduction[amount_of_blocks, block_size](input_matrix, summed_array_red)

summed_array_red = np.zeros(arraySize)

times = []

for i in range(10):
    input_matrix_2 = input_matrix_3.copy()
    summed_array_red = np.zeros(arraySize)
    start = time.time()
    sum_axis_matrix_reduction[amount_of_blocks, block_size](input_matrix_2, summed_array_red)
    total = time.time() - start
    times.append(total)

print("GPU RESULTS REDUCTION")
print(summed_array_red)
print(times)
print(np.average(times))
