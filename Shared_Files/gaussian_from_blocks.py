from helper_functions.file_io import *
from morph_filter import get_block 
from gaussian_fitter import fit_gaussian
import matplotlib.pyplot as plt

def par_frame_fit_as_dict(par_frame_fit):

    offset = par_frame_fit[0]
    amplitude = par_frame_fit[1]
    theta = par_frame_fit[2]
    x0 = par_frame_fit[3]
    y0 = par_frame_fit[4]
    sigmax = par_frame_fit[5]
    sigmay = par_frame_fit[6]

    output = {'offset': offset, 'amplitude': amplitude, 'theta': theta, 'x0': x0, 'y0': y0, 'sigmax': sigmax, 'sigmay': sigmay}

    return(output)

def main():

    file_names, path = get_directory_input()

    file_name = file_names[0]

    image = read_fits_file(os.path.join(path, file_name))

    img_height, img_width = image.shape
    
    n_rows = 12
    n_cols = 12
    n_blocks = n_rows * n_cols

    block_height = img_height // n_rows
    block_width = img_width // n_cols

    square_length = 10
    half = square_length // 2
    
    for row_idx in range(0, 3):
        for col_idx in range(0, 3):

            subframe = get_block(image, row_idx, col_idx, BLOCK_H=block_height, BLOCK_W=block_width)

            # find brightest PSF per subframe
            psf_idx = np.argmax(subframe)
            psf_y, psf_x = np.unravel_index(psf_idx, subframe.shape)

            results = fit_gaussian(image, square_length, int(psf_x), int(psf_y))
            par_frame_fit = results[2]
            fitted_gaussian = results[1]

            new_frame = np.zeros(subframe.shape)

            new_frame[psf_y-half:psf_y+half, psf_x-half:psf_x+half] = fitted_gaussian

            plt.imshow(subframe)
            plt.imshow(new_frame)

            plt.show()





if __name__ == "__main__":
    main()