from im_filtering.morph_filter_square import get_block
from im_filtering.morph_filter_square import set_block
from im_filtering.helper_functions.file_io import *

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    im_shape = (1944,2592)

    x_tiles = 3
    y_tiles = 2

    x_size = im_shape[1]//x_tiles
    y_size = im_shape[0]//y_tiles

    composite_output = np.zeros(im_shape)
    for y_ind in range(0,y_tiles):
        for x_ind in range(0,x_tiles):

            file_names, path_input = get_directory_input(message = f"Please input path for tile at {x_ind+1}, {y_ind+1}: ",allow_folder=False)
            frame = read_fits_file(os.path.join(path_input, file_names[0]))

            block = get_block(frame, y_ind, x_ind, y_size, x_size)

            # plt.figure()
            # plt.imshow(frame)

            # plt.figure()
            # plt.imshow(block)
            # plt.show()

            set_block(composite_output, y_ind, x_ind, block, y_size, x_size)

    write_fits_file(composite_output, os.path.join(path_input, "composite_by_tile_output.fits"))



