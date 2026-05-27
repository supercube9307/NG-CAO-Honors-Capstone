from astropy.io import fits
import numpy as np
import os


if __name__ == '__main__':


    filename = r"C:\Users\antho\Videos\NG\PSF_Characterization\p_0-0\pos_ZeroZero-2.Raw"
    for i in range(5):

        raw_file = filename #+ str(i+1) +'.raw'
        raw_imarray = np.fromfile(raw_file, dtype='int16')
        reshaped_raw_imarray = np.reshape(raw_imarray, (1944, 2592))
        fits_file = filename.split('.')[0]+'.fits'
        hdu = fits.ImageHDU(reshaped_raw_imarray)
        prim = fits.PrimaryHDU()
        hdul = fits.HDUList([prim, hdu])
        hdul.writeto(fits_file, overwrite=True)