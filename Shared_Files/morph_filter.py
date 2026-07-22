"""
Generated from: https://claude.ai/chat/8127038a-0298-4861-b22a-5ed76877934e

load_in_image.py

Python equivalent of Copy_of_LoadInImage.m

Loads a stack of FITS calibration frames, takes the first frame, splits it
into a 12x12 grid of 162x216 blocks, applies a white top-hat morphological
filter to each block (using a footprint_rectangle structuring element whose size varies
by block, mirroring the MATLAB switch/case logic), and reassembles + displays
the filtered image.

Requires: numpy, astropy, scikit-image, matplotlib
    pip install numpy astropy scikit-image matplotlib
"""

import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from skimage.morphology import white_tophat
from skimage.morphology.footprints import footprint_rectangle
from helper_functions.file_io import *

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
IMGS_FOLDER = "Shared_Files/Bad_Pixel_Calibration_Frames/averaged"
FILE_TYPE = ".fits"
FITS_EXTNAME = "Image"  # extension name used by fitsread(fpath(x), "Image")

IMG_HEIGHT = 1944
IMG_WIDTH = 2592
BLOCK_H = 162
BLOCK_W = 216
N_BLOCK_ROWS = IMG_HEIGHT // BLOCK_H  # 12
N_BLOCK_COLS = IMG_WIDTH // BLOCK_W  # 12
N_BLOCKS = N_BLOCK_ROWS * N_BLOCK_COLS  # 144

# Block index (1-based, MATLAB-style) -> structuring element size
STREL_SIZE_FOR_BLOCK = {}


def _register(indices, size):
    for i in indices:
        STREL_SIZE_FOR_BLOCK[i] = size


_register(
    [1, 2, 24, 36, 38, 39, 48, 50, 52, 64, 69, 76, 81, 86, 88, 93, 108, 111,
     120, 132, 133, 134],
    8,
)
_register(
    [3, 10, 11, 12, 14, 15, 26, 27, 35, 40, 47, 53, 54, 55, 56, 57, 59, 65,
     66, 67, 68, 71, 77, 78, 80, 83, 89, 90, 91, 92, 95, 98, 100, 101, 102,
     104, 105, 110, 112, 135, 143, 144],
    7,
)
_register(
    [4, 5, 7, 8, 9, 16, 17, 19, 20, 21, 22, 23, 28, 29, 31, 32, 33, 34, 41,
     42, 44, 45, 46, 58, 70, 79, 82, 94, 103, 106, 107, 113, 114, 116, 117,
     118, 119, 122, 123, 136, 140, 141, 142],
    6,
)
_register([6, 18, 30, 43, 115, 124, 130, 131, 137, 138, 139], 5)
_register([125, 126, 128, 129], 4)
_register([127], 3)
_register(
    [13, 25, 51, 60, 62, 63, 72, 74, 84, 87, 96, 99, 109, 121],
    9,
)
_register([37, 75, 97], 10)
_register([49, 61, 73, 85], 11)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def get_block(image, row_idx, col_idx, BLOCK_H = BLOCK_H, BLOCK_W = BLOCK_W):
    """Extract block (row_idx, col_idx) using 1-based MATLAB-style indexing."""
    r0 = (row_idx) * BLOCK_H
    c0 = (col_idx) * BLOCK_W
    return image[r0:r0 + BLOCK_H, c0:c0 + BLOCK_W]


def set_block(dest, row_idx, col_idx, block, BLOCK_H = BLOCK_H, BLOCK_W = BLOCK_W):
    """Write block (row_idx, col_idx) into dest using 1-based indexing."""
    r0 = (row_idx) * BLOCK_H
    c0 = (col_idx) * BLOCK_W
    dest[r0:r0 + BLOCK_H, c0:c0 + BLOCK_W] = block

def morph_filter_image(source_image) -> np.ndarray:

    # ---- Segmentation start: split into 12x12 grid of blocks ----
    blocks = np.zeros((BLOCK_H, BLOCK_W, N_BLOCKS))

    z = 0
    block_positions = []  # (row_idx, col_idx) for each z, in fill order
    for x in range(0, N_BLOCK_ROWS):
        for y in range(0, N_BLOCK_COLS):
            blocks[:, :, z] = get_block(source_image, x, y)
            block_positions.append((x, y))
            z += 1

    # ---- Apply white top-hat filtering with per-block structuring element ----
    for x in range(0, N_BLOCKS):
        size = STREL_SIZE_FOR_BLOCK.get(x)
        if size is not None:
            blocks[:, :, x] = white_tophat(blocks[:, :, x], footprint_rectangle([size, size]))

    # ---- Reassemble filtered blocks into the full image ----
    filtered_image = np.zeros((IMG_HEIGHT, IMG_WIDTH))
    for z, (x, y) in enumerate(block_positions):
        set_block(filtered_image, x, y, blocks[:, :, z])

    return(filtered_image)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # ---- Load in the images ----
    
    file_names, path = get_directory_input()

    for file_name in file_names: 

        file_path = os.path.join(path, file_name)

        source_image = read_fits_file(file_path)

        filtered_image = morph_filter_image(source_image)

        # ---- Display ----

        plt.figure(figsize=(IMG_HEIGHT, IMG_WIDTH,'px'))
        plt.imshow(filtered_image, cmap="gray", vmin=np.min(filtered_image), vmax=np.max(filtered_image))
        plt.axis("off")
        # plt.savefig("local_data/morph_filter_frames/python_morph_filter.png",bbox_inches='tight',)
        plt.colorbar()
        plt.title("Filtered Image (auto-scaled)")

        plt.show()


if __name__ == "__main__":
    main()
