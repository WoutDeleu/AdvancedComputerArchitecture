import math
import time

import numpy as np
from numba import cuda

n = 1000
summed_array_red = np.zeros(n)


@cuda.jit
def gpu_reduce(input_arr, output_arr):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    #sA = cuda.shared.array(shape=4, dtype=numba.float32)
    #sA[tid] = input_arr[i]
    #cuda.syncthreads()

    s=1
    while s < int(cuda.blockDim.x):
        if tid % (2 * s) == 0:
            input_arr[tid][bid] += input_arr[tid + s][bid]
        cuda.syncthreads()
        s *= 2

    if tid == 0:
        output_arr[cuda.blockIdx.x] = input_arr[0][bid]


block_size = math.ceil((n*n)/1024)
threads = math.ceil((n*n)/block_size)

group_size = n
block_size = n

array_2d = np.random.randint(1,3, size=(n,n))
correct_result = np.sum(array_2d, axis=0)

print(array_2d)
print(correct_result)
gpu_reduce[group_size, block_size](array_2d, summed_array_red)
print(summed_array_red)

block_size2 = math.ceil(n/1024)
threads2 = math.ceil(n/block_size2)



print(np.all(correct_result == summed_array_red))