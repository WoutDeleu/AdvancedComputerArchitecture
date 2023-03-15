import numpy as np
from PIL import Image
from numba import cuda

img = Image.open("sheldon.jpeg")
img_arr = np.array(img)


#inversion = 255-R, 255-G, 255-B
def inverse(img_arr, output):
    x = cuda.blockIdx.x
    y = cuda.blockIdx.y
    idx = x + y * cuda.gridDim.x


    for i in range():



    for (int i = 0; i < Channels; i++){
        Image[idx + i] = 255 - Image[idx + i];
    }


print(img_arr)