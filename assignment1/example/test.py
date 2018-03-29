from skimage import io
import numpy as np

im1=io.imread('1.pgm')
im2=io.imread('2.pgm')

x = np.array(im1)
y = np.array(im2)
z = abs(x/100 - y/100)
z=z*100;
SAD = np.sum(z)
SSD = np.sum(z**2)
print(SAD,SSD)