from astropy.io import fits
import numpy as np
import os
from directory_input import get_directory_input

transform = 1

if __name__ == '__main__':

    file_names, path_input = get_directory_input("raw")

    for filename in file_names:
    
        print("Converting " + filename)

        if transform == 0:
            for i in range(0,9999):
                raw_file1 = filename #+ str(i)
                for j in range(0,99): 
                    raw_file1 = raw_file1 + '-' + str(j) + '.raw'

                    if os.path.isfile(raw_file1):
                        raw_imarray = np.fromfile(raw_file1, dtype='uint16')
                        reshaped_raw_imarray = np.reshape(raw_imarray, (1944,2592))
                        fits_file = raw_file1.split('.')[0]+'.fits'
                        hdu = fits.ImageHDU(reshaped_raw_imarray)
                        prim = fits.PrimaryHDU()
                        hdul = fits.HDUList([prim,hdu])
                        hdu.writeto(fits_file, overwrite=True)

        if transform == 1:
            
            raw_file1 = filename #+ str(j) + '.Raw'
            if os.path.isdir(path_input):
                raw_file1 = path_input+"/"+raw_file1

            raw_imarray = np.fromfile(raw_file1, dtype='uint16')
            reshaped_raw_imarray = np.reshape(raw_imarray, (1944,2592))
            
            fits_file = raw_file1.split('.')[0]+'.fits'
            
            hdu = fits.ImageHDU(reshaped_raw_imarray)
            prim = fits.PrimaryHDU()
            hdul = fits.HDUList([prim,hdu])
            hdul.writeto(fits_file, overwrite=True)

        if transform == 2:
            for i in range(0,9999):
                raw_file1 = filename + str(i) + '-'
                for j in range(0,40):
                    raw_file2 = raw_file1 + str(j) + '.Raw'

                    if os.path.isfile(raw_file2):
                        raw_imarray = np.fromfile(raw_file2, dtype='uint16')
                        reshaped_raw_imarray = np.reshape(raw_imarray, (int(1944/2 - 12),int(2592/2),2))
                        fits_file = raw_file2.split('.')[0]+'.fits'
                        hdu = fits.ImageHDU(reshaped_raw_imarray[:,:,0])
                        prim = fits.PrimaryHDU()
                        hdul = fits.HDUList([prim,hdu])
                        hdu.writeto(fits_file, overwrite=True)