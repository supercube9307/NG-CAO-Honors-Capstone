import matplotlib.pyplot as plt

import im_filtering.bad_pixel_calibration as bp
import im_filtering.non_uniformity_calibration as nuc
import im_filtering.morph_filter_ellipse as morph
from im_filtering.helper_functions.file_io import *

def dark_subtract(image: np.ndarray) -> np.ndarray:

    dark_names, path_input = get_directory_input(message="Input path to dark frame .fits file: ")
    dark_frame = read_fits_file(os.path.join(path_input, dark_names[0]))

    try:
        return(image-dark_frame)
    except:
        return(dark_frame)

def combined_filtering(image: np.ndarray) -> np.ndarray:

    dark_sub = dark_subtract(image)
    bp_corr = bp.bp_correction(dark_sub)
    nuc_corr = nuc.apply_calibration(bp_corr)
    morph_filtered = morph.morph_filter_image_ellipse(nuc_corr)

    return(morph_filtered)

def main():

    file_names, path_input = get_directory_input()
    file_path = os.path.join(path_input, file_names[0])
    image = read_fits_file(file_path)

    filtered_image = combined_filtering(image)

    plt.figure()
    plt.imshow(image)
    plt.title("Original Image")

    plt.figure()
    plt.imshow(filtered_image)
    plt.title("Filtered Image")
    plt.colorbar
    plt.show()    

    filtered_file_path = file_path.split(".")[0] + "_filtered.fits"
    write_fits_file(filtered_image, filtered_file_path)

if __name__ == "__main__":
    main()