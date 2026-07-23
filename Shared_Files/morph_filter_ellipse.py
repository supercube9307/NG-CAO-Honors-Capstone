import numpy as np
from skimage.morphology import white_tophat
import matplotlib.pyplot as plt
import json
from skimage.morphology.footprints import footprint_rectangle

from helper_functions.file_io import *
from morph_filter_square import get_block
from morph_filter_square import set_block

def ellipse_from_parameters(par, square_length, show_gauss=False) -> np.ndarray:

    amplitude = par['amplitude']
    theta = par['theta']
    x0 = par['x0']
    y0 = par['y0']
    sigmax = par['sigmax']
    sigmay = par['sigmay']

    xp, yp = np.meshgrid(np.arange(0, square_length), np.arange(0, square_length))

    fitted_gaussian = amplitude * np.exp(
        -(np.cos(theta) * (xp - x0) - np.sin(theta) * (yp - y0)) ** 2 / (2 * sigmax ** 2)
        - (np.sin(theta) * (xp - x0) + np.cos(theta) * (yp - y0)) ** 2 / (2 * sigmay ** 2)
    )

    if show_gauss:
        plt.figure()
        plt.title(f"Gaussian")
        plt.imshow(fitted_gaussian)
        plt.colorbar()

    strel = (fitted_gaussian > 1).astype(int)

    return(strel)

def get_pars_list(strel_file_path = 'local_data/morph_filter/strel_parameters.json') -> tuple:

    pars_list = []
    square_size = 0
    n_rows = 0
    n_cols = 0

    try:
        with open(strel_file_path) as strel_file:
            output_dict = json.load(strel_file)

            pars_list = output_dict['pars_list']
            square_size = output_dict['square_length']
            n_rows = output_dict['n_rows']
            n_cols = output_dict['n_cols']

    except FileNotFoundError:

        message = f"""
{os.path.join(os.getcwd(), strel_file_path)} was not found.
Please run 'ellipse_footprints_from_blocks.py' to generate list of morphological filter footprints."""
        
        print(message)

    return(pars_list, square_size, n_rows, n_cols)

def show_block(block, filtered_block, row_idx, col_idx, verbose = False):

    if verbose == True:
        plt.figure()
        plt.title(f"Block ({row_idx}, {col_idx})")
        plt.imshow(block)
        plt.colorbar()

        plt.figure()
        plt.title(f"Filtered Block ({row_idx}, {col_idx})")
        plt.imshow(filtered_block)
        plt.colorbar()

        plt.show()

def write_morph_filter_output(image, file_path: str):

    response_verified = False

    while not(response_verified):

        response = input("Write to file? [y/n]: ")

        response = response.lower()[0]

        if response == 'y' or response == 'n':
            response_verified = True

    file_path_filtered = file_path.split('.')[0] + '_morph_filtered.fits'

    if response == 'y':

        write_fits_file(image, file_path_filtered)

def morph_filter_image(image,show_blocks=False, show_gauss=False) -> np.ndarray:

    filtered_image = np.zeros(image.shape)

    img_height, img_width = image.shape

    pars_list, square_length, n_rows, n_cols = get_pars_list()

    block_height = img_height // n_rows
    block_width = img_width // n_cols

    if len(pars_list) == 0:
        return
    
    for row_idx in range(0, n_rows):
        for col_idx in range(0, n_cols):

            print(f"Analyzing block ({row_idx+1}, {col_idx+1})")

            block_idx = row_idx * n_rows + col_idx

            pars = pars_list[block_idx]

            strel = ellipse_from_parameters(pars, square_length, show_gauss)

            block = get_block(image, row_idx, col_idx, BLOCK_H=block_height, BLOCK_W=block_width)
            try:
                filtered_block = white_tophat(block, strel)
            except ValueError:
                filtered_block = block

            set_block(filtered_image, row_idx, col_idx, filtered_block, BLOCK_H=block_height, BLOCK_W=block_width)

            show_block(block, filtered_block, row_idx, col_idx, verbose=show_blocks)

    return(filtered_image)



def main():
    
    file_names, path = get_directory_input()

    file_name = file_names[0]

    file_path = os.path.join(path, file_name)

    image = read_fits_file(file_path)

    filtered_image = morph_filter_image(image) # show_blocks = True, show_gauss = True)

    plt.figure()
    plt.title(f"Image")
    plt.imshow(image)
    plt.colorbar()

    plt.figure()
    plt.title(f"Filtered Image")
    plt.imshow(filtered_image)
    plt.colorbar()

    plt.show()

    write_morph_filter_output(image, file_path)
    

if __name__ == '__main__':
    main()