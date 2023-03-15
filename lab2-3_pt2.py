import numpy as np
from PIL import Image
from numba import cuda

img = Image.open("sheldon.jpeg")
img_arr = np.array(img)
output_arr = np.zeros_like(img_arr)

#inversion = 255-R, 255-G, 255-B
def inverse(img_arr, output):
    x = cuda.blockIdx.x
    y = cuda.blockIdx.y
    idx = x + y * cuda.gridDim.x

    for i in range(100):
        output[idx + i] = 255 - img_arr[idx + i];


inverse(img_arr, output_arr)
print(output_arr)