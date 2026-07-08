from astropy.io import fits
import numpy as np
import os
from directory_input import get_directory_input

if __name__ == '__main__':
    
    file_names, path_input = get_directory_input()

    for file in file_names:
        try 

    hdu = fits.ImageHDU(averages)
    prim = fits.PrimaryHDU()
    hdul_ouptut = fits.HDUList([prim,hdu])
    
    average_filepath = path_input+"\\Average_Output.fits"
    hdul_ouptut.writeto(average_filepath, overwrite=True)

    print("Wrote to file "+average_filepath)