import matplotlib.pyplot as plt

import im_filtering.bad_pixel_calibration as bp
import im_filtering.non_uniformity_calibration as nuc
import im_filtering.morph_filter_ellipse as morph
from im_filtering.helper_functions.file_io import *

def combined_filtering(image):

    bp_corr = bp.bp_correction(image)

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