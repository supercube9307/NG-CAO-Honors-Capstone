from astropy.io import fits
import numpy as np
import os
from directory_input import get_directory_input
        
def average_fits(file_names, path_input):
    
    averages = []
    for file in file_names:
    
        print("Converting " + file)
        file_path =  path_input+"/"+file

        with fits.open(file_path) as hdul:
            x_length = int(hdul[1].header["NAXIS1"])
            y_length = int(hdul[1].header["NAXIS2"])

            if len(averages) == 0:
                averages = np.zeros((y_length,x_length))

            averages = np.add(averages,hdul[1].data)

        averages /= len(file_names)
    
    return averages


if __name__ == '__main__':
    
    file_names, path_input = get_directory_input()

    averages = average_fits(file_names, path_input)

    hdu = fits.ImageHDU(averages)
    prim = fits.PrimaryHDU()
    hdul_ouptut = fits.HDUList([prim,hdu])
    
    average_filepath = path_input+"\\Average_Output.fits"
    hdul_ouptut.writeto(average_filepath, overwrite=True)

    print("Wrote to file "+average_filepath)