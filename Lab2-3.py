import math

import numpy as np
from numba import cuda

sizeArray = 10
number_blocks_x = math.ceil(sizeArray / 1024)
number_blocks_y = math.ceil(sizeArray / 1024)

#amount of threads, can also be int
block_size = 32, 32

#can also be int
amount_of_blocks = number_blocks_x, number_blocks_y


@cuda.jit
def sum_columns_matrix_atomic(matrix, summed):
    x, y = cuda.grid(2)
    cuda.atomic.add(summed, x, matrix[x][y])
    # todo: make dynamic


@cuda.jit
def sum_columns_matrix_reduction(matrix, summed):
    x = cuda.blockIdx.x
    y = cuda.threadIdx.x

    i = 1
    while i <= max:
        if y % (2 * i) == 0:
            matrix[x][y] += matrix[x][y + i]
        i *= 2
    cuda.syncthreads()
    if cuda.threadIdx.x == 0:
        summed[x] = matrix[x][0]


summed_array_atomic = np.zeros(sizeArray)
summed_array_reduction = np.zeros(sizeArray)
input_matrix = np.random.randint(10, size=(sizeArray, sizeArray))
sum_columns_matrix_atomic[amount_of_blocks, block_size](input_matrix, summed_array_atomic)
#sum_columns_matrix_reduction[amount_of_blocks, block_size](input_matrix, summed_array_reduction)

print(input_matrix)
print()
print(summed_array_atomic)
#print(summed_array_reduction)