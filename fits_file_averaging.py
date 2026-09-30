from astropy.io import fits
import numpy as np
import os
from im_filtering.helper_functions.file_io import *
        
def average_fits(file_names, path_input) -> np.ndarray:
    
    frames = []
    for file in file_names:
    
        print("Including " + file)
        file_path =  os.path.join(path_input,file)

        frame = read_fits_file(file_path)
        frames.append(frame)

    total = np.zeros(frames[0].shape)

    for frame in frames:
        total += frame

    average = total / len(frames)
    
    return(average)

def longest_common_substring(strings) -> str:
    default_output = "output"
    if not strings:
        return default_output
    
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
                
    return default_output

if __name__ == '__main__':
    
    file_names, path_input = get_directory_input(allow_folder=True)

    average = average_fits(file_names, path_input)

    trimmed_filenames = []
    for file_name in file_names:
        trimmed_filenames.append(file_name.split(".")[0])
    
    average_filename = longest_common_substring(trimmed_filenames)+"_average.fits"
    average_filepath = os.path.join(path_input,average_filename)

    write_fits_file(average, average_filepath)