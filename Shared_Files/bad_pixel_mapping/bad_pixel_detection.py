from astropy.io import fits
import numpy as np
import os



if __name__ == '__main__':
    operation_mode = input("Please select 0 for dark frames or 1 for flat frames: ")
    file_path = input("Input path to file: ")
    
    with fits.open(file_path) as hdul:
        file_data = hdul[1].data
        print(file_data)

    # if operation_mode == 0:
    #     dark_bp_detection()