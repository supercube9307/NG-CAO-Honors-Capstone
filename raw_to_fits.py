from astropy.io import fits
import numpy as np
import os
from im_filtering.helper_functions.file_io import *


if __name__ == '__main__':

    file_names, path_input = get_directory_input(expected_file_type="raw", allow_folder=True)

    for filename in file_names:

        print("Converting " + filename)

        if os.path.isdir(path_input):
            filename = os.path.join(path_input, filename)

        try:
            raw_imarray = np.fromfile(filename, dtype='uint16')
            reshaped_raw_imarray = np.reshape(raw_imarray, (1944,2592))
        except ValueError:
            raw_imarray = np.fromfile(filename, dtype='uint8')
            reshaped_raw_imarray = np.reshape(raw_imarray, (1944,2592))

        
        fits_file_name = filename.split('.')[0]+'.fits'
        
        write_fits_file(reshaped_raw_imarray, fits_file_name, file_output_required=True)