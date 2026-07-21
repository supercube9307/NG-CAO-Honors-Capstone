from astropy.io import fits
import numpy as np
import os
from fits_file_handling.file_io import *
from Shared_Files.fits_file_handling.file_io import get_directory_input


if __name__ == '__main__':

    file_names, path_input = get_directory_input("raw")

    for filename in file_names:

        print("Converting " + filename)

        if os.path.isdir(path_input):
            filename = os.path.join(path_input, filename)

        raw_imarray = np.fromfile(filename, dtype='uint16')
        reshaped_raw_imarray = np.reshape(raw_imarray, (1944,2592))
        
        fits_file = filename.split('.')[0]+'.fits'
        
        image = fits.ImageHDU(reshaped_raw_imarray)
        prim = fits.PrimaryHDU()
        hdul = fits.HDUList([prim,image])
        hdul.writeto(fits_file, overwrite=True)