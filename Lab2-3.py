import numpy as np
from numba import cuda

sizeArray = 4
numberBlocks = 1
numberThreads = [sizeArray, sizeArray]


@cuda.jit
def sum_columns_matrix_atomic(matrix, summed):
    x, y = cuda.grid(2)
    cuda.atomic.add(summed, x, matrix[x][y])


summed_array = np.zeros(sizeArray)
matrix = np.random.randint(10, size=(sizeArray, sizeArray))
sum_columns_matrix_atomic[numberBlocks, numberThreads](matrix, summed_array)

print(matrix)
print(summed_array)
