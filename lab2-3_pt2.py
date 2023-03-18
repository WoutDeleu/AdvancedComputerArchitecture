import math
import numpy as np
from PIL import Image
from numba import cuda

threads_per_block = (8, 8, 8)

img = Image.open("sheldon.jpeg")
img_arr = np.array(img)
output_arr = np.zeros_like(img_arr)

input = cuda.to_device(img_arr)
output = cuda.device_array_like(output_arr)

number_blocks_x = math.floor((img_arr.shape[1] + threads_per_block[1] - 1) / threads_per_block[1])
number_blocks_y = math.floor((img_arr.shape[0] + threads_per_block[0] - 1) / threads_per_block[0])

blocks_per_grid = (number_blocks_y, number_blocks_x, 1)

#inversion = 255-R, 255-G, 255-B
@cuda.jit
def inverse(img_arr, inverted_arr):
    x = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    y = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
    z = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z

    if x < img_arr.shape[1] and y < img_arr.shape[0] and z < img_arr.shape[2]:

        #row, column, depth
        inverted_arr[y, x, z] = 255 - img_arr[y, x, z]


inverse[blocks_per_grid, threads_per_block](img_arr, output)
output_arr = output.copy_to_host()
output_img = Image.fromarray(output_arr)
output_img.save("inverted_sheldon.jpeg")