import numpy as np
from im_filtering.helper_functions.file_io import *

class image_x_y:
    def __init__(self, x_value, y_array):
        self.x_value = x_value
        self.y_array = y_array
        self.y_avg = np.average(y_array)

def get_x_value(file_name):

    x_value = None

    while x_value == None:

        input_text = input(f"Please input x value for {file_name}: ")

        try:
            x_value = float(input_text)
        except:
            print("Please input a valid float")

    return(x_value)
    
def compute_calibration(dark_frame, flat_frame):

    y_array_diff = np.subtract(flat_frame.y_array, dark_frame.y_array)
    y_avg_diff = flat_frame.y_avg - dark_frame.y_avg 

    C_gain_array = np.divide(y_avg_diff, y_array_diff)
    C_offset_array = np.subtract(dark_frame.y_avg, np.multiply(C_gain_array, dark_frame.y_array))

    return(C_gain_array, C_offset_array)

def apply_calibration(image: np.ndarray, gain_array_path="local_data/2_point_NUC/C_gain_array.fits", offset_array_path="local_data/2_point_NUC/C_offset_array.fits"):

    try:
        gain_array = read_fits_file(gain_array_path)
        offset_array = read_fits_file(offset_array_path)
    except FileNotFoundError:
        print("Calibration data not found in " + os.getcwd() + ". Regenerating " + gain_array_path + " and " + offset_array_path)
        generate_calibration(file_output_required=True)
        gain_array = read_fits_file(gain_array_path)
        offset_array = read_fits_file(offset_array_path)

    filtered_image = np.multiply(gain_array, image) + offset_array

    return(filtered_image)

def generate_calibration(file_output_required=False):

    frame_types = ["dark", "flat"]
    images = {}
    x_values = {}
    for frame_type in frame_types:
        message = "\nPlease input path to .fits file containing master " + frame_type + " frame: "

        file_names, path_input = get_directory_input(message = message, allow_folder=False)
        file_path =  os.path.join(path_input,file_names[0])

        images[frame_type] = read_fits_file(file_path)
        x_values[frame_type] = get_x_value(file_names[0])

    dark_frame = image_x_y(x_values["dark"],images["dark"])
    flat_frame = image_x_y(x_values["flat"],images["flat"])

    output_arrays = compute_calibration(dark_frame, flat_frame)

    print("Generated Gain Corrections and Offset Corrections")

    output_file_names = ["local_data/2_point_NUC/C_gain_array.fits", "local_data/2_point_NUC/C_offset_array.fits"]
    for file_name, array in zip(output_file_names, output_arrays):
        write_fits_file(array, file_name, file_output_required=file_output_required)

def main():
    generate_calibration()

if __name__ == "__main__":
    main()

