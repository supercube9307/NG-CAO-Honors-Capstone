from astropy.io import fits
import numpy as np
import os

if __name__ == '__main__':

    path_verified = False
    file_paths = [];
    while path_verified == False:

        path_input = input("Input path to folder containing .fits files: ")

        if not(os.path.exists(path_input)):
            print("Please select a valid file path")
            continue
        
        if os.path.isfile(path_input):
            print("Please input a path to a folder")
            continue

        if os.path.isdir(path_input):
            print("Parsing Folder")
            for file in os.listdir(path_input):
                file_paths.append(file)

        for file in file_paths:
            if file.split(".")[-1] != "fits":
                print("Ingoring " + file)
                file_paths.remove(file)

        if len(file_paths) == 0:
            print("Please select a file or folder with .fits files")
            continue
        else:
            break
  

    averages = []

    for filename in file_paths:
    
        print("Converting " + filename)
        file_path =  path_input+"\\"+filename

        with fits.open(file_path) as hdul:
            x_length = int(hdul[1].header["NAXIS1"])
            y_length = int(hdul[1].header["NAXIS2"])

            if len(averages) == 0:
                averages = np.zeros((y_length,x_length))

            averages = np.add(averages,hdul[1].data)
            
            #for x_index in range(x_length):
            #    print(x_index)
            #    for y_index in range(y_length):
            #        value = hdul[1].data[y_index][x_index]
            #        averages += value

        averages /= len(file_paths)

    hdu = fits.ImageHDU(averages)
    prim = fits.PrimaryHDU()
    hdul_ouptut = fits.HDUList([prim,hdu])
    
    average_filepath = path_input+"\\Average_Output.fits"
    hdu.writeto(average_filepath, overwrite=True)

    print("Wrote to file "+average_filepath)