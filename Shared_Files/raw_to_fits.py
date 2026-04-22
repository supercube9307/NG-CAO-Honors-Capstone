from astropy.io import fits
import numpy as np
import os

transform = 1

if __name__ == '__main__':

    path_verified = False
    file_paths = [];
    while path_verified == False:

        path_input = input("Input path to file or folder: ")

        if not(os.path.exists(path_input)):
            print("Please select a valid file path")
            continue
        
        if os.path.isfile(path_input):
            print("Parsing File")
            file_paths.append(path_input)

        elif os.path.isdir(path_input):
            print("Parsing Folder")
            for file in os.listdir(path_input):
                file_paths.append(file)

        for file in file_paths:
            if file.split(".")[-1] != "raw":
                print("Ingoring " + file)
                file_paths.remove(file)

        if len(file_paths) == 0:
            print("Please select a file or folder with .raw")
        else:
            break

    for filename in file_paths:
    
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
                raw_file1 = path_input+"\\"+raw_file1

            raw_imarray = np.fromfile(raw_file1, dtype='uint16')
            reshaped_raw_imarray = np.reshape(raw_imarray, (1944,2592))
            
            fits_file = raw_file1.split('.')[0]+'.fits'
            
            hdu = fits.ImageHDU(reshaped_raw_imarray)
            prim = fits.PrimaryHDU()
            hdul = fits.HDUList([prim,hdu])
            hdu.writeto(fits_file, overwrite=True)

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

        # print(fits_file)