import math
import time

import numpy as np
from PIL import Image
from numba import cuda

#z is RGB!
threads_per_block = (8, 8, 3)

img = Image.open("appel-artikel.jpg")
image_array = np.array(img)
output_array = np.zeros_like(image_array)


print(image_array)

number_blocks_x = math.ceil((image_array.shape[1]) / threads_per_block[1])
number_blocks_y = math.ceil((image_array.shape[0]) / threads_per_block[0])
number_blocks_z = math.ceil((image_array.shape[2]) / threads_per_block[2])

blocks_per_grid = (number_blocks_y, number_blocks_x, number_blocks_z)

# inversion = 255-R, 255-G, 255-B
@cuda.jit
def inverse(image_array, inverted_arr):
    x_init = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    y_init = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
    z_init = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z

    for z in range(z_init, image_array.shape[2], cuda.blockDim.z * cuda.gridDim.z):
        for y in range(y_init, image_array.shape[0], cuda.blockDim.y * cuda.gridDim.y):
            for x in range(x_init, image_array.shape[1], cuda.blockDim.x * cuda.gridDim.x):
                if x < image_array.shape[1] and y < image_array.shape[0] and z < image_array.shape[2]:
                    # row, column, depth
                    inverted_arr[y, x, z] = 255 - image_array[y, x, z]



inverse[blocks_per_grid, threads_per_block](image_array, output_array)

times = []

for i in range(10):
    start = time.time()
    inverse[blocks_per_grid, threads_per_block](image_array, output_array)
    total = time.time() - start
    times.append(total)

print(times)
print(np.average(times))

output_img = Image.fromarray(output_array)
output_img.save("inverted-appel.jpg")
