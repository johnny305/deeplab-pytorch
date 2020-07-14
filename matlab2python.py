# encoding: utf-8
# Author: SunJackson 
# URL: https://github.com/SunJackson/fcn_val/blob/master/VOCevalseg.py
import numpy as np
from PIL import Image

colors_map = [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
        [0, 0, 255],
        [252, 230, 201]
]

def matlab_imread(im_path):
    im = Image.open(im_path)  # Replace with your image name here
    # moded L to mode P
    im = im.convert("P", palette=colors_map)
    indexed = np.array(im)  # Convert to NumPy array to easier access

    # Get the colour palette
    palette = im.getpalette()
    # Determine the total number of colours
    num_colours = len(palette) // 3

    # Determine maximum value of the image data type
    max_val = float(np.iinfo(indexed.dtype).max)

    # Create a colour map matrix
    map = np.array(palette).reshape(num_colours, 3) / max_val
    return  indexed, map
