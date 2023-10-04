# -*- coding: utf-8 -*-

__author__ = "Christine S. Isaksen"
__email__ = "christine.sande.isaksen@nmbu.no"

import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# Read image
image = io.imread('gingerbreads.jpg')

def threshold(image, th=None):
    """Returns a binarised version of given image, thresholded at given value.

    Binarises the image using a global threshold `th`. Uses Otsu's method
    to find optimal threshold value if the threshold variable is None. The
    returned image will be in the form of an 8-bit unsigned integer array
    with 255 as white and 0 as black.

    Parameters:
    -----------
    image : np.ndarray
        Image to binarise. If this image is a colour image then the last
        dimension will be the colour value (as RGB values).
    th : numeric
        Threshold value. Uses Otsu's method if this variable is None.

    Returns:
    --------
    binarised : np.ndarray(dtype=np.uint8)
        Image where all pixel values are either 0 or 255.
    """
    shape = np.shape(image)
    binarised = np.zeros([shape[0], shape[1]], dtype=np.uint8)

    if len(shape) == 3:
        image = image.mean(axis=2)
    elif len(shape) > 3:
        raise ValueError('Must be at 2D image')

    if th is None:
        th = otsu(image)

    binary = image > th
    binarised = (binary + binarised) * 255

    return binarised


def histogram(image):
    """
    Returns the image histogram with 256 bins.
    """
    shape = np.shape(image)
    histogram = np.zeros(256)


    if len(shape) == 3:
        image = image.mean(axis=2)
    elif len(shape) > 3:
        raise ValueError('Must be at 2D image')

    for i in range(shape[0]):
        for j in range(shape[1]):
            pixval = int(image[i,j])
            histogram[pixval] += 1

    return histogram



def otsu(image):
    """
    Finds the optimal threshold value of given image using Otsu's method.
    """
    hist = histogram(image)
    total_pixels = np.sum(hist)
    sum_intensity = np.sum([i * hist[i] for i in range(256)])
    sum_background = 0
    weight_background = 0
    max_var = 0
    th = 0

    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue
        weight_foreground = total_pixels - weight_background
        
        if weight_foreground == 0:
            break
        sum_background += t * hist[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_intensity - sum_background) / weight_foreground
        var_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if var_between > max_var:
            max_var = var_between
            th = t

    return th


"""
Plots original picture, histogram and new picture
"""
new_image = threshold(image)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax0, ax1, ax2 = axes.ravel()

ax0.imshow(image, cmap=plt.cm.gray)
ax0.set_title('Original Image')

hist = histogram(image)
ax1.plot(hist, color='black')
ax1.set_title('Histogram')
ax1.set_xlim([0, 256])

ax2.imshow(new_image, cmap=plt.cm.gray)
ax2.set_title('Thresholded Image')

plt.savefig('mandatory_exercise_1.jpg')
plt.show()
