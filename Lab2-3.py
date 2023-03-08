import numba
import numpy as np
from numba import cuda

sizeArray = 8
numberBlocks = 1
numberThreads = [sizeArray, sizeArray]

@cuda.jit
def sum_columns_matrix(matrix, summed):
    column_id = cuda.grid(2)
    print(column_id)

summed = 0
matrix = np.random.randint(10, size=(sizeArray, sizeArray))
sum_columns_matrix[numberBlocks, numberThreads](matrix, summed)

print(matrix)
print(summed)
