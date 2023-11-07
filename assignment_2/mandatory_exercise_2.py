# -*- coding: utf-8 -*-

from skimage import io, color, morphology, feature, segmentation, measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
import numpy as np

# Reads the image then resizes it to only capture the chocolates
image = io.imread('IMG_2754_nonstop_alltogether.JPG')
image = image[200:3500, 200:5500]

# Converted image to greyscale
grey_image = color.rgb2gray(image)

# Reduces noise with thresholding
threshold_value = 0.5
binary_image = grey_image > threshold_value

# Morphed the image so that the nonstops are monochromatic 
# (without the shine)
binary_image_inverted = np.invert(binary_image)
filled_image = ndimage.binary_fill_holes(binary_image_inverted).astype(bool)

# Morphed the image to remove chocolate crumbs
erode_image = morphology.erosion(filled_image, morphology.disk(15))
dilate_image = morphology.dilation(erode_image, morphology.disk(15))

# Using scikit's watershed method
distance = ndimage.distance_transform_edt(dilate_image)
local_max = feature.peak_local_max(distance, footprint=morphology.disk(15), min_distance=50)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(local_max.T)] = True
markers, _ = ndimage.label(mask)
labels = segmentation.watershed(-distance, markers, mask=dilate_image)

# Region Properties Calculation (area, perimeter etc.)
region = measure.regionprops(labels)

# Create and plot squares that distinguish between Nonstop, broken Nonstops and m&m
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image)

colored_patches = {}

for props in region:
    circularity = 4 * np.pi * (props.area / props.perimeter ** 2)
    roundness = 4 * props.area / (np.pi * props.axis_major_length**2)

    if roundness > 0.83 or circularity < 0.83:
        color = 'white'
    else:
        color = 'black'

    minr, minc, maxr, maxc = props.bbox
    colored_patches[color] = ax.add_patch(patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                            fill=False, edgecolor=color, linewidth=2))


plt.title('Nonstop and m&m separated')
plt.legend([colored_patches['white'], colored_patches['black']], ['Non stop', 'm&m'])     
plt.show()