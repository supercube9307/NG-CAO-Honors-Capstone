from astropy.io import fits
import numpy as np
import os
from im_filtering.helper_functions.file_io import *

def composite_fits(file_names, path_input):
    images = []

    composite = None

    for file_name in file_names:
        image = read_fits_file(os.path.join(path_input, file_name))
        images.append(image)

        if len(images) == 1:
            composite = np.zeros(images[0].shape)

        composite = np.maximum(image, composite)

    return(composite)
        

if __name__ == '__main__':
    
    file_names, path_input = get_directory_input()

    composite = composite_fits(file_names, path_input)
    composite_filepath = os.path.join(path_input,"Composite_Output.fits")

    hdu = fits.ImageHDU(composite)
    prim = fits.PrimaryHDU()
    hdul_ouptut = fits.HDUList([prim,hdu])
    
    hdul_ouptut.writeto(composite_filepath, overwrite=True)

    print("Wrote to file "+composite_filepath)