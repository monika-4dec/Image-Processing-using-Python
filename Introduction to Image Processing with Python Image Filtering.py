# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 13:12:20 2023

@author: Monika201103
"""
"""Introduction to Image Processing with Python — Image Filtering
Edge Detection and Other Morphological Operators for Beginners"""

""" #importing the required Python Libraries"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage import img_as_uint
from scipy.signal import convolve2d

"""#define a simple edge detection kernel"""

kernel_edgedetection = np.array([[-1, -1, -1],
                                 [-1, 8.5, -1],
                                 [-1, -1, -1]])
imshow(kernel_edgedetection, cmap = 'gray');

morph = imread('Sample.jpg')
"""To apply the edge detection kernel to our image we simply have to use the 
convolve2d function in SciPy. But before that, we must first convert our 
image into greyscale (remember that we are applying a 2 Dimensional kernel)."""
plt.figure(num=None, figsize=(8, 6), dpi=80)
morph_gray = rgb2gray(morph)
imshow(morph_gray);

"""To apply the kernel, we can simply use the convolve2d function in SciPy."""

conv_im1 = convolve2d(morph_gray, kernel_edgedetection)
imshow(abs(conv_im1) , cmap='gray');

def edge_detector(image):
    f_size = 15
    morph_gray = rgb2gray(image)
    kernels = [np.array([[-1, -1, -1],
                         [-1, i, -1],
                        [-1, -1, -1]]) for i in range(2,10,1)]    
    
    titles = [f'Edge Detection Center {kernels[k][1][1]}' for k in
              range(len(kernels))]
    
    fig, ax = plt.subplots(2, 4, figsize=(17,12))
    
    for n, ax in enumerate(ax.flatten()):
        ax.set_title(f'{titles[n]}', fontsize = f_size)
        ax.imshow(abs(convolve2d(morph_gray, kernels[n])) , 
                  cmap='gray')
        ax.set_axis_off()
        
    fig.tight_layout()

# Horizontal Sobel Filter
h_sobel = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])
# Vertical Sobel Filter
v_sobel = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])
fig, ax = plt.subplots(1, 2, figsize=(17,12))
ax[0].set_title(f'Horizontal Sobel', fontsize = 15)
ax[0].imshow(h_sobel, cmap='gray')
ax[0].set_axis_off()
ax[1].set_title(f'Vertical Sobel', fontsize = 15)
ax[1].imshow(v_sobel , cmap='gray')
ax[1].set_axis_off()

fig, ax = plt.subplots(1, 2, figsize=(17,12))
ax[0].set_title(f'Horizontal Sobel', fontsize = 15)
ax[0].imshow(abs(convolve2d(morph_gray, h_sobel)), cmap='gray')
ax[0].set_axis_off()
ax[1].set_title(f'Vertical Sobel', fontsize = 15)
ax[1].imshow(abs(convolve2d(morph_gray, v_sobel)) , cmap='gray')
ax[1].set_axis_off()