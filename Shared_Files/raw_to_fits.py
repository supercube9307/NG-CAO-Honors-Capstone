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
            print("Please select a file or folder with .raw files")
        else:
            break

    averages = []

    for filename in file_paths:
    
        print("Converting " + filename)


        with fits.open(filename) as hdul:
            x_length = hdul[1].header["NAXIS1"]
            y_length = hdul[1].header["N"]

            if len(averages) == 0:
                averages = np.empty(x_length,y_length)
            
            for x_index in range(x_length):
                for y_index in range(y_length):
                    averages += hdul[1].data[y_index][x_index]

        averages /= len(file_paths)

    print(averages)