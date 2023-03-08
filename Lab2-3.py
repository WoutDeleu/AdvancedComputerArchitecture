import math

import numpy as np
from numba import cuda

sizeArray = 10000
number_blocks_x = math.ceil(sizeArray / 1024)
number_blocks_y = math.ceil(sizeArray / 1024)
block_size = [32, 32]
amount_of_blocks = [number_blocks_x, number_blocks_y]


@cuda.jit
def sum_columns_matrix_atomic(matrix, summed):
    x, y = cuda.grid(2)
    cuda.atomic.add(summed, x, matrix[x][y])


summed_array = np.zeros(sizeArray)
input_matrix = np.random.randint(10, size=(sizeArray, sizeArray))
sum_columns_matrix_atomic[amount_of_blocks, block_size](input_matrix, summed_array)

print(input_matrix)
print()
print(summed_array)
