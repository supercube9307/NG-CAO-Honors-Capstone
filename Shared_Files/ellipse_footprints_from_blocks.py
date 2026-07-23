from helper_functions.file_io import *
from morph_filter_square import get_block 
from gaussian_fitter import fit_gaussian
import matplotlib.pyplot as plt
import json

def parameters_as_dict(par):

    amplitude = par[1]
    theta = par[2]
    x0 = par[3]
    y0 = par[4]
    sigmax = par[5]
    sigmay = par[6]

    dict = {'amplitude':amplitude, 'theta':theta, 'x0':x0, 'y0':y0, 'sigmax':sigmax, 'sigmay':sigmay}

    return(dict)

def show_residuals(subframe, fitted_gaussian, half, psf_x, psf_y, block_idx, verbose = False):
        
    if verbose == True:

        new_frame = np.zeros(subframe.shape)
        new_frame_padded = new_frame.copy() + 5
        new_frame_padded = np.pad(new_frame_padded, half)
        new_frame_padded -= 5

        psf_y_padded = psf_y + half
        psf_x_padded = psf_x + half

        new_frame_padded[psf_y_padded-half:psf_y_padded+half, psf_x_padded-half:psf_x_padded+half] = fitted_gaussian
        new_frame = new_frame_padded[half:subframe.shape[0]+half,half:subframe.shape[1]+half]

        plt.figure()
        plt.imshow(subframe)
        plt.title(f"Subframe {block_idx}")
        plt.colorbar()

        plt.figure()
        plt.imshow(new_frame)
        plt.title(f"New Frame {block_idx}")
        plt.colorbar()

        plt.figure()
        plt.imshow(subframe - new_frame)
        plt.title(f"subframe - new_frame {block_idx}")
        plt.colorbar()

        plt.show()

def generate_strel_parameters(image, n_rows = 12, n_cols = 12) -> dict:

    img_height, img_width = image.shape

    block_height = img_height // n_rows
    block_width = img_width // n_cols

    square_length = 10
    half = square_length // 2

    pars_list = []
    
    for row_idx in range(0, n_rows):
        for col_idx in range(0, n_cols):

            block_idx = n_rows*row_idx + col_idx
            print(f"Analyzing block ({row_idx+1}, {col_idx+1})")

            subframe = get_block(image, row_idx, col_idx, BLOCK_H=block_height, BLOCK_W=block_width)

            # ignore PSFs too close to edge
            subframe = subframe[half: block_height - half, half: block_width - half]
            subframe = np.pad(subframe, half)

            # find brightest PSF per subframe
            psf_idx = np.argmax(subframe)
            psf_y, psf_x = np.unravel_index(psf_idx, subframe.shape)

            results = fit_gaussian(subframe, square_length, int(psf_x), int(psf_y), verbose=False)
            parameters = results[2]

            fitted_gaussian = results[1] - parameters[0] #remove offset from Gaussian
            show_residuals(subframe, fitted_gaussian, half, psf_x, psf_y, block_idx, verbose=False)

            par_dict = parameters_as_dict(parameters)

            pars_list.append(par_dict)

    output_dict = {'square_length':square_length, 'pars_list':pars_list, 'n_rows':n_rows, 'n_cols':n_cols}

    return(output_dict)

def main():

    file_names, path = get_directory_input()

    file_name = file_names[0]

    image = read_fits_file(os.path.join(path, file_name))

    output_dict = generate_strel_parameters(image)

    output_str = json.dumps(output_dict, indent=2)

    output_path = 'local_data/morph_filter/strel_parameters.json'
    with open(output_path,"w") as output_file:
        output_file.write(output_str)

    print(f'Wrote to: {os.path.join(os.getcwd(), output_path)}')



if __name__ == "__main__":
    main()