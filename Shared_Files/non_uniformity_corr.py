import numpy as np
import os
from astropy.io import fits
import fits_file_handling.directory_input as di

class image_x_y:
    def __init__(self, x_value, y_array):
        self.x_value = x_value
        self.y_array = y_array
        self.y_avg = np.average(y_array)

def get_x_value():

    x_value = None

    while x_value == None:

        input_text = input(f"Please input x value for {file_name}: ")

        try:
            x_value = float(input_text)
        except:
            print("Please input a valid float")

    return(x_value)

def compile_image_x_y(x_value):
    
    with fits.open(file_path) as hdul:

        file_data = hdul[0].data

        return(image_x_y(x_value,file_data))
    
def compute_calibration(images):

    y_min = None
    y_max = None
    for image in images:

        if y_min == None:
            y_min = image.y_avg

        if y_max == None:
            y_max = image.y_avg

        y_min = min(y_min,image.y_avg)
        y_max = max(y_max,image.y_avg)

    for image in images:

        if image.y_avg == y_max:
            bright_frame = image

        if image.y_avg == y_min:
            dark_frame = image

    y_array_diff = np.subtract(bright_frame.y_array, dark_frame.y_array)
    y_avg_diff = y_max - y_min

    C_gain_array = np.divide(y_avg_diff, y_array_diff)
    C_offset_array = np.subtract(y_min, np.multiply(C_gain_array, dark_frame.y_array))

    return(C_gain_array, C_offset_array)



if __name__ == "__main__":

    file_names, path_input = di.get_directory_input(message="Input path to folder containing .fits files: ")

    images = []
    for file_name in file_names:

        x_value = get_x_value()

        file_path = path_input+"/"+file_name

        images.append(compile_image_x_y(x_value))

    output_arrays = compute_calibration(images)

    print("Generated Gain Corrections and Offset Corrections")

    output_file_names = ["C_gain_array.csv", "C_offset_array.csv"]
    for x in range(0,2):
        print("\nFor "+output_file_names[x]+":")
        di.output_csv_file(output_arrays[x], file_name=output_file_names[x])

