import numpy as np
from skimage.morphology import white_tophat
import matplotlib.pyplot as plt
import json

from helper_functions.file_io import *
from morph_filter_square import get_block
from morph_filter_square import set_block

def ellipse_from_parameters(par, show_ellipse=False, n_sigma = 2) -> np.ndarray:

    theta = par['theta']
    sigmax = par['sigmax']
    sigmay = par['sigmay']

    max_sigma = np.sqrt(int(max(n_sigma * sigmax, n_sigma * sigmay)))
    square_length = 2*max_sigma+1
    center = max_sigma

    xp, yp = np.meshgrid(np.arange(0, square_length), np.arange(0, square_length))
    xp -= center
    yp -= center

    xp_theta = xp * np.cos(theta) + yp * np.sin(theta)
    yp_theta = -xp * np.sin(theta) + yp * np.cos(theta)

    ellipse_theta = (xp_theta) ** 2 / (sigmax * n_sigma) + (yp_theta) ** 2 / (sigmay * n_sigma)

    strel_theta = (ellipse_theta <= 1).astype(int)

    if show_ellipse:
        plt.figure()
        plt.title(f"Ellipse")
        plt.imshow(strel_theta)
        plt.colorbar()

    # if strel_theta.size > 10000:
    #     strel_theta = np.zeros([5,5])

    return(strel_theta)

def get_pars_list(strel_file_path = 'local_data/morph_filter/strel_parameters.json') -> tuple:

    pars_list = []
    n_rows = 0
    n_cols = 0

    try:
        with open(strel_file_path) as strel_file:
            output_dict = json.load(strel_file)

            pars_list = output_dict['pars_list']
            n_rows = output_dict['n_rows']
            n_cols = output_dict['n_cols']

    except FileNotFoundError:

        message = f"""
{os.path.join(os.getcwd(), strel_file_path)} was not found.
Please run 'ellipse_footprints_from_blocks.py' to generate list of morphological filter footprints."""
        
        print(message)

    return(pars_list, n_rows, n_cols)

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
    file_path_filtered = file_path_filtered.split(os.path.sep)[-1]

    if response == 'y':

        write_fits_file(image, file_path_filtered)

def morph_filter_image_ellipse(image,show_blocks=False, show_ellipse=False) -> np.ndarray:

    filtered_image = np.zeros(image.shape)

    img_height, img_width = image.shape

    pars_list, n_rows, n_cols = get_pars_list()

    if len(pars_list) == 0:
        # ask user to generate strel parameters file
        return

    block_height = img_height // n_rows
    block_width = img_width // n_cols
    
    for row_idx in range(0, n_rows):
        for col_idx in range(0, n_cols):

            print(f"Analyzing block ({row_idx+1}, {col_idx+1})")

            block_idx = row_idx * n_rows + col_idx

            pars = pars_list[block_idx]

            strel = ellipse_from_parameters(pars, show_ellipse = show_ellipse)

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

    filtered_image = morph_filter_image_ellipse(image) #, show_blocks = True, show_ellipse = True)

    if type(filtered_image) == type(None):
        # ask user to generate strel parameters file
        return

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