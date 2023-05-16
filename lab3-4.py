import math
import time
from timeit import timeit
import matplotlib.pyplot as plt

import numpy as np
from numba import cuda

# 1024 threads per block
block_size = 32, 32

def matrix_multiplication_CPU(A, B, resulting_matrix):
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                resulting_matrix[i][j] += A[i][k] * B[k][j]


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
def matrix_multiplication_GPU_shared_memory(A, B, resulting_matrix):
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



size = 1
timing_cpu = {}
timing_cpu_np = {}
timing_gpu_naive = {}
timing_gpu_shared = {}

while size < 1050:

    print(size)
    number_blocks_x = math.ceil(size / block_size[0])
    number_blocks_y = math.ceil(size / block_size[1])
    amount_of_blocks = number_blocks_y, number_blocks_x

    A = np.random.randint(10, size=(size, size))
    B = np.random.randint(10, size=(size, size))
    resulting_matrix = np.zeros((size, size))

    # cpu_np = lambda: np.matmul(A, B)
    # timing_cpu_np[size] = timeit(cpu_np, number=20)

    # if(size < -1):
    #     cpu = lambda: matrix_multiplication_CPU(A, B, resulting_matrix)
    #     timing_cpu[size] = timeit(cpu, number=20)
    #     resulting_matrix = np.zeros((size, size))

    a_dev = cuda.to_device(A)
    b_dev = cuda.to_device(B)
    res_dev = cuda.to_device(resulting_matrix)

    gpu_naive = lambda: matrix_multiplication_GPU[amount_of_blocks, block_size](a_dev, b_dev, res_dev)
    # timing_gpu_naive[size] = timeit(gpu_naive, number=20)

    # precompiling
    # gpu_naive()
    # cuda.synchronize()

    start = time.time()
    for i in range(20):
        gpu_naive()
        cuda.synchronize()
    duration = time.time() - start
    timing_gpu_naive[size] = duration

    resulting_matrix_2 = np.zeros((size, size))
    res_dev_2 = cuda.to_device(resulting_matrix_2)

    gpu_shared = lambda: matrix_multiplication_GPU_shared_memory[amount_of_blocks, block_size](a_dev, b_dev, res_dev_2)
    # timing_gpu_shared[size] = timeit(gpu_shared, number=20)

    # gpu_shared()
    # cuda.synchronize()

    start = time.time()
    for i in range(20):
        gpu_shared()
        cuda.synchronize()
    duration = time.time() - start
    timing_gpu_shared[size] = duration

    size *= 2

# print(timing_cpu)
print(timing_cpu_np)
print(timing_gpu_naive)
print(timing_gpu_shared)

sizes = list(timing_gpu_naive.keys())
cpu_times = list(timing_cpu.values())
cpu_np_times = list(timing_cpu_np.values())
gpu_naive_times = list(timing_gpu_naive.values())
gpu_shared_times = list(timing_gpu_shared.values())

# plt.plot(sizes, cpu_np_times, label='CPU Numpy')
plt.plot(sizes, gpu_naive_times, label='GPU Repetitive memory access')
plt.plot(sizes, gpu_shared_times, label='GPU Shared Memory')
plt.legend()

plt.title('Multiplication duration for different implementations')
plt.xlabel('Square matrix order')
plt.ylabel('Time (ms)')
plt.yscale('log')

plt.show()



#  CHAINING
size = 1

timing_no_migration = {}
timing_migration = {}

while size < 2100:
    number_blocks_x = math.ceil(size / block_size[0])
    number_blocks_y = math.ceil(size / block_size[1])
    amount_of_blocks = number_blocks_y, number_blocks_x

    A = np.random.randint(10, size=(size, size))
    B = np.random.randint(10, size=(size, size))
    C = np.random.randint(10, size=(size, size))
    D = np.random.randint(10, size=(size, size))

    resulting_matrix_AB = np.zeros((size, size))
    resulting_matrix_ABC = np.zeros((size, size))
    resulting_matrix_ABCD = np.zeros((size, size))
    gpu_normal = lambda: (
        matrix_multiplication_GPU[amount_of_blocks, block_size](A, B, resulting_matrix_AB),
        matrix_multiplication_GPU[amount_of_blocks, block_size](resulting_matrix_AB, C, resulting_matrix_ABC),
        matrix_multiplication_GPU[amount_of_blocks, block_size](resulting_matrix_ABC, D, resulting_matrix_ABCD)
    )

    gpu_normal()
    timing_migration[size] = timeit(gpu_normal, number=20)

    a_dev = cuda.to_device(A)
    b_dev = cuda.to_device(B)
    c_dev = cuda.to_device(C)
    d_dev = cuda.to_device(D)
    resulting_matrix = np.zeros((size, size))
    res_dev_AB = cuda.to_device(resulting_matrix)
    res_dev_ABC = cuda.to_device(resulting_matrix)
    res_dev_ABCD = cuda.to_device(resulting_matrix)
    gpu_no_migration = lambda: (
        matrix_multiplication_GPU[amount_of_blocks, block_size](a_dev, b_dev, res_dev_AB),
        matrix_multiplication_GPU[amount_of_blocks, block_size](res_dev_AB, c_dev, res_dev_ABC),
        matrix_multiplication_GPU[amount_of_blocks, block_size](res_dev_ABC, d_dev, res_dev_ABCD)
    )

    # precompiling
    gpu_no_migration()
    cuda.synchronize()

    start = time.time()
    for i in range(20):
        gpu_no_migration()
        cuda.synchronize()
    duration = time.time() - start
    timing_no_migration[size] = duration

    size *= 2


sizes = list(timing_no_migration.keys())
times_without_migration = list(timing_no_migration.values())
times_with_migration = list(timing_migration.values())


plt.plot(sizes, times_without_migration, label='No migration overhead')
plt.plot(sizes, times_with_migration, label='Migration overhead')
plt.legend()

plt.title('Memory migration impact on chained square matrix multiplications (ABCD)')
plt.xlabel('Square matrix order')
plt.ylabel('Time (ms)')
# plt.yscale('log')

plt.show()
