from astropy.io import fits
import numpy as np
import os

import bp_util.fixedbp as fixedbp
from fits_file_handling.directory_input import get_directory_input

def analyze_file(file_path, operation_mode):

        with fits.open(file_path) as hdul:
            file_data = hdul[0].data
        
        if operation_mode == 0:
            sd_threshold = int(input("Input Standard Deviation(s) above which pixels should be discarded: "))
            dark_bp_locs = fixedbp._compute_fixedbp_from_dark(file_data,sd_threshold)
            pretty_output(dark_bp_locs)
            
        if operation_mode == 1:
            percentage_threshold = int(input("Input Percentage below which pixels should be discarded (0-100): "))/100
            flat_bp_locs = fixedbp._compute_fixedbp_from_flat(file_data,percentage_threshold,32)
            pretty_output(flat_bp_locs)

def pretty_output(bool_2d):
    
    output_list = get_bool_2d_locs(bool_2d)

    if len(output_list) == 0:
        print("No Bad Pixels found within threshold")
        return
    
    for bp_loc in output_list:
        print(f"Bad Pixel found at: {bp_loc}")

    file_output_required = input("Would you like to save these locations to a file? (Y/N): ").lower()
    if file_output_required:
        output_csv_file(output_list)

def output_csv_file(output_list):

    with open("output.csv", "w") as output_file:

        output_text = ""
        for bp_loc in output_list:

            bp_text = str(bp_loc[0])+","+str(bp_loc[1])+"\n"
            output_text += bp_text

        output_file.write(output_text)

    print("File 'output.csv' created in " + os.getcwd())


def get_bool_2d_locs(bool_2d):

    x_len = len(bool_2d[0])
    y_len = len(bool_2d)
    output_list = []

    for y_index in range(y_len):
        for x_index in range(x_len):

            if bool_2d[y_index][x_index] == True:
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

