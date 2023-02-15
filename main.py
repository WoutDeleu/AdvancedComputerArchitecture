import time
import numpy as np
from numba import cuda

N = 100
input_arr = np.arange(N)
output_arr = np.zeros_like(input_arr)

input_cpu = input_arr
output_cpu = output_arr

# Define kernel ( GPU ) function ...
@cuda.jit
def flip (input_arr, output_arr):
    for i in range (0, N):
        print(i)



# Call kernel function ...
# Time it
start = time.time()
#flip[1,N](input_arr, output_arr)
cuda.synchronize()
total = time.time() - start
print(total)




start = time.time()
for i in range (0,int(N/2)):
    temp = input_cpu[i]
    output_cpu[i] = input_cpu[N-i-1]
    output_cpu[N-1-i] = temp
total = time.time() - start
print("CPU RESULTS:")
print(total)
print(input_cpu)
print(output_cpu)

