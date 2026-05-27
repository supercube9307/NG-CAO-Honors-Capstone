from astropy.io import fits
import numpy as np
import os

import bp_util.fixedbp as fixedbp
from fits_file_handling.directory_input import get_directory_input

def analyze_file(file_path, operation_mode):

        with fits.open(file_path) as hdul:
            file_data = hdul[0].data
        
        if operation_mode == 0:
            dark_bp_locs = fixedbp._compute_fixedbp_from_dark(file_data,5)
            print(pretty_output(dark_bp_locs))
            
        if operation_mode == 1:
            flat_bp_locs = fixedbp._compute_fixedbp_from_flat(file_data,0.1,16)
            print(pretty_output(flat_bp_locs))

def pretty_output(bool_2d):

    x_len = len(bool_2d)
    y_len = len(bool_2d[0])
    output_list = []

    for x_index in range(x_len):
        for y_index in range(y_len):

            if bool_2d[x_index][y_index] == True:
                output_list.append([x_index,y_index])

    return(output_list)

if __name__ == '__main__':

    operation_mode = -1
    while operation_mode == -1:
        text_input = input("Please select 0 for dark frames or 1 for flat frames: ")
        if text_input == "0" or text_input == "1":
            operation_mode = int(text_input)
            break
        print("Invalid Input")

    file_names, path_input = get_directory_input()
    
    for file in file_names:
        file_path =  path_input+"/"+file
        analyze_file(file_path, operation_mode)

