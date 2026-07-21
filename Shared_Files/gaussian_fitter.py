"""
Python conversion of gaussian_fitter.m

https://claude.ai/chat/821bf681-6e20-4bc5-a676-f6cc65fd18ae

Notes on the conversion
------------------------
- MATLAB's LMFnlsq (a third-party Levenberg-Marquardt solver) is replaced
  with scipy.optimize.least_squares(..., method='lm'), which implements
  the same underlying algorithm (MINPACK's lmdif), with FunTol/XTol
  mapped to ftol/xtol.
- MATLAB uses 1-based, inclusive-end array indexing; the subframe crop
  below has been converted to 0-based, exclusive-end Python slicing that
  selects the identical set of pixels.
- MATLAB globals (frame_to_fit, xp, yp, ifit, iter) are replaced with
  ordinary local variables that get passed explicitly into
  rotated_gaussian / rotated_gaussian_residuals.
- imagesc(...) -> plt.imshow(..., origin='upper') with a colorbar.
- The helper functions clnShow and nPeaks (defined but unused after the
  `return` in the original .m file) are included at the bottom in
  Python form for completeness/parity, faithfully preserving the
  original logic (including an apparent no-op comparison in clnShow's
  `highNufs` line, which is always true as written in the source).
"""

import numpy as np
from scipy.io import loadmat
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from astropy.io import fits
import os

import fits_file_handling.file_io as file_io
from rotated_gaussian import *


def matlab_range_to_slice(a, b):
    """Convert an inclusive 1-based MATLAB range a:b to a Python slice."""
    return slice(a - 1, b)

def fit_gaussian(image: np.ndarray, square_length: int, psf_x: int, psf_y: int) -> tuple:

    # frame = loadmat('J.mat')
    # frame = loadmat('Shared_Files/dot_grid.mat')
    # image = frame['fits']
    half = round(square_length / 2) 

    row_slice = matlab_range_to_slice(psf_x - half, psf_x + half)
    col_slice = matlab_range_to_slice(psf_y - half, psf_y + half)

    subframe = image[row_slice, col_slice].astype(float)
    # subframe = frame['J'][matlab_range_to_slice(1025-half,1025+half),
    #                       matlab_range_to_slice(2185-half,2185+half)]

    frame_to_fit = subframe  # col 27 (index 26) of subframe holds the PSF

    initial_pars = np.array([300, 15000, 0, square_length / 2, square_length / 2, 1, 1], dtype=float)

    xp, yp = np.meshgrid(np.arange(1, square_length + 2), np.arange(1, square_length + 2))

    ifit = 1

    result = least_squares(
        rotated_gaussian_residuals,
        initial_pars,
        args=(xp, yp, frame_to_fit),
        kwargs={'ifit': ifit, 'verbose': True},
        method='lm',
        ftol=1e-7,
        xtol=1e-7,
    )
    par_frame_fit = result.x

    _, fitted_gaussian = rotated_gaussian(par_frame_fit, xp, yp, frame_to_fit, ifit=ifit, verbose=False)
    print(par_frame_fit)

    return(subframe, fitted_gaussian, par_frame_fit, xp, yp)

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

def show_results(subframe: np.ndarray, fitted_gaussian, par_frame_fit: tuple, xp, yp):

    plt.figure()
    plt.imshow(subframe)
    plt.colorbar()

    plt.figure()
    # look at residual
    plt.imshow(subframe - fitted_gaussian)
    plt.colorbar()

    square_length = subframe.shape[0]
    x0 = par_frame_fit[3]  # ellipse centre coordinates
    y0 = par_frame_fit[4]
    sigmax = abs(par_frame_fit[5]) # horizontal radius
    sigmay = abs(par_frame_fit[6]) # vertical radius

    # rotate the ellipse according to the fitted orientation angle
    theta = par_frame_fit[2]
    X = np.cos(theta) * (xp - x0) - np.sin(theta) * (yp - y0)
    Y = np.sin(theta) * (xp - x0) + np.cos(theta) * (yp - y0)

    # get the pixels that fall closest to the ellipses (shade in the regions)
    ellipse_pixels1 = (X ** 2 / sigmax ** 2 + Y ** 2 / sigmay ** 2 <= 1)
    ellipse_pixels2 = (X ** 2 / (2 * sigmax) ** 2 + Y ** 2 / (2 * sigmay) ** 2 <= 1)
    ellipse_pixels3 = (X ** 2 / (3 * sigmax) ** 2 + Y ** 2 / (3 * sigmay) ** 2 <= 1)

    plt.figure()
    plt.imshow(fitted_gaussian)
    plt.colorbar()

    plt.figure()
    subframe1 = subframe.copy()
    subframe1[ellipse_pixels1] = np.inf  # color the ellipses to show up on figure
    plt.imshow(subframe1)
    plt.colorbar()

    plt.figure()
    subframe2 = subframe.copy()
    subframe2[ellipse_pixels2] = np.inf
    plt.imshow(subframe2)
    plt.colorbar()

    plt.figure()
    subframe3 = subframe.copy()
    subframe3[ellipse_pixels3] = np.inf
    plt.imshow(subframe3)
    plt.colorbar()

    # normalized power spectral density of PSF
    PSF_col = subframe[:, round(square_length/2)]  # column 14 (1-based) of subframe for this particular PSF
    L = PSF_col.shape[0] - 1  # length of signal (minus 1 to make it even, if needed)
    Fs = 1  # sampling frequency: 1 per pixel is all we have
    f = Fs / L * np.arange(0, L / 2 + 1)  # frequency domain, positive range only

    Y_fft = np.abs(np.fft.fft(PSF_col))
    # divide by length of signal (fft scaling), multiply by 2 (amplitude split
    # across +/- frequencies), then square for power spectral amplitude
    Y_fft = (2 * Y_fft / L) ** 2
    Ypos = Y_fft[:int(np.floor(L / 2)) + 1]

    plt.figure()
    plt.semilogy(f, Ypos / np.sum(np.abs(Ypos)))

    plt.show()

def main():

    psf_x = 930
    psf_y = 810

    file_names, path = file_io.get_directory_input()    

    file_path = os.path.join(path, file_names[0])

    image = file_io.read_fits_file(file_path)

    square_length = 10

    results = fit_gaussian(image, square_length, psf_x, psf_y)

    par_frame_fit = results[2]

    show_results(*results)

if __name__ == '__main__':
    main()