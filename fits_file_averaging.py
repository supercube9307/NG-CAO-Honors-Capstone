from astropy.io import fits
import numpy as np
import os
from im_filtering.helper_functions.file_io import *
        
def average_fits(file_names, path_input) -> np.ndarray:
    
    averages = []
    for file in file_names:
    
        print("Including " + file)
        file_path =  path_input+"/"+file

        with fits.open(file_path) as hdul:
            x_length = int(hdul[1].header["NAXIS1"]) # type: ignore
            y_length = int(hdul[1].header["NAXIS2"]) # type: ignore

            if len(averages) == 0:
                averages = np.zeros((y_length,x_length))

            averages = np.add(averages,hdul[1].data) # type: ignore

        averages /= len(file_names)
    
    return averages

def longest_common_substring(strings) -> str:
    if not strings:
        return ""
    
    # 1. Start with the shortest string to minimize iterations
    shortest = min(strings, key=len)
    length = len(shortest)
    
    # 2. Check substrings from largest to smallest window size
    for width in range(length, 0, -1):
        for start in range(length - width + 1):
            substr = shortest[start:start + width]
            
            # 3. Verify if it exists in all other strings
            if all(substr in s for s in strings):
                return substr
                
    return ""


if __name__ == '__main__':
    
    file_names, path_input = get_directory_input(allow_folder=True)

    average = average_fits(file_names, path_input)

    trimmed_filenames = []
    for file_name in file_names:
        trimmed_filenames.append(file_name.split(".")[0])
    
    average_filename = longest_common_substring(trimmed_filenames)+"average.fits"
    average_filepath = os.path.join(path_input,average_filename)

    write_fits_file(average, average_filepath)