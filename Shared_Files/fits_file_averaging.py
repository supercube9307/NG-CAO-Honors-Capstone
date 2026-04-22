from astropy.io import fits
import numpy as np
import os

def get_directory_input():

    file_names = []
    path_verified = False
    while path_verified == False:

        path_input = input("Input path to folder containing .fits files: ")

        if not(os.path.exists(path_input)):
            print("Please select a valid file path")
            continue
        
        if os.path.isfile(path_input):
            print("Please input a path to a folder")
            continue

        print("Parsing Folder")
        for file in os.listdir(path_input):
            file_names.append(file)

        for file in file_names:
            if file.split(".")[-1] != "fits":
                print("Ingoring " + file)
                file_names.remove(file)

        if len(file_names) == 0:
            print("Please select a file or folder with .fits files")
            continue
        else:
            return(file_names, path_input)

        
def average_fits(file_names, path_input):
    
    averages = []
    for file in file_names:
    
        print("Converting " + file)
        file_path =  path_input+"\\"+file

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
    hdu.writeto(average_filepath, overwrite=True)

    print("Wrote to file "+average_filepath)