import math
from misc_functions import (sumMat, maxk, items_from_indices, flatten, createMini,
    Pars2GaussMini, makeUMap, calcSigEst, calcSigEst2, find_ones, makeI2, lst_sq_fit)
import numpy as np
import sympy
from scipy.interpolate import RectBivariateSpline, LSQBivariateSpline, SmoothBivariateSpline
from scipy.spatial import KDTree
from scipy.special import j1
from scipy.optimize import curve_fit, fsolve, minimize, Bounds
from nPeaks import nPeaks2
from center_of_distortion import enhance
from fit_cents_to_grid import get_fov_odd_square

def nCentroids_grid(pix_corr_x, pix_corr_y, M_in, n, m, num_std_dev, ceil, Method, SNRthresh, box_size, num_gauss_std_dev, bright_x, bright_y, Gauss_sigma_local):
    '''
    first will find the n highest peaks in the graph using nPeaks.
    Then for each peak, we find draw a box around that peak and calculate the centroid of the star in the box.
    This makes our list of centroids.
    '''


    y = nPeaks2(M_in, n, m, num_std_dev, ceil, box_size, num_gauss_std_dev, bright_x, bright_y)
    #y = nPeaks(M_in, n, m)  #could take a long time
    print('nPeaks done.  Peaks found:  ', len(y))
    cents = []
    cent_x = []
    cent_y = []
    cents_grid = np.zeros_like(M_in)
    #for i in range(n):
    if Method != 'FFT' and Method != 'FFT2':
        centroid = FGCentroid3(pix_corr_x, pix_corr_y, M_in, y[0][1], y[0][2], box_size, Method, SNRthresh, bright_x, bright_y) #brightest first one done separately
        #centroid = FGCentroid2(M_in, y[0][1], y[0][2], box_size, Method, SNRthresh) #brightest first one done separately
        temp_box_size = box_size
        while np.isnan(float(centroid['y'])) or np.isnan(float(centroid['x'])) or centroid['y'] < 0 or centroid['x'] < 0 or centroid['y'] > M_in.shape[0]-1 or centroid['x'] > M_in.shape[1]-1:
            temp_box_size -= 1
            centroid = FGCentroid3(pix_corr_x, pix_corr_y, M_in, y[0][1], y[0][2], temp_box_size, Method, SNRthresh, bright_x, bright_y)
            #centroid = FGCentroid2(M_in, y[0][1], y[0][2], temp_box_size, Method, SNRthresh)
        if np.round(float(centroid['y'])) >= 0 and np.round(float(centroid['y'])) <= cents_grid.shape[0]-1 and \
            np.round(float(centroid['x'])) >= 0 and np.round(float(centroid['x'])) <= cents_grid.shape[1]-1:
            cents_grid[(int(np.round(float(centroid['y']))), int(np.round(float(centroid['x']))))] = 1
        cents.append([float(centroid['y']), float(centroid['x'])])
        cent_x.append(float(centroid['x']))
        cent_y.append(float(centroid['y']))

        sigmax = []
        sigmay = []
        offset_list = []

        points = np.array([(y[i][2], y[i][1]) for i in range(len(y))])
        kdtree = KDTree(points)
        nine_dist, nine_ind = kdtree.query(points[0], k=9)
        #for i in range(1, 9): # to get the avg sigma x and sigma y for the next 8 PSFs, assumed to be close to brightest one and close to cod
        for i in nine_ind[1:]:
            row = y[i][1]
            col = y[i][2]
            centroid = FGCentroid3(pix_corr_x, pix_corr_y, M_in, row, col, box_size, Method, SNRthresh, bright_x, bright_y)
            #centroid = FGCentroid2(M_in, row, col, box_size, Method, SNRthresh)
            #centroid ={'y': row, 'x': col} This was used just to see how well the peaks themselves did for centroids
            temp_box_size = box_size
            while np.isnan(float(centroid['y'])) or np.isnan(float(centroid['x'])) or centroid['y'] < 0 or centroid['x'] < 0 or centroid['y'] > M_in.shape[0]-1 or centroid['x'] > M_in.shape[1]-1:
                temp_box_size -= 1
                centroid = FGCentroid3(pix_corr_x, pix_corr_y, M_in, row, col, temp_box_size, Method, SNRthresh, bright_x, bright_y)
                #centroid = FGCentroid2(M_in, row, col, temp_box_size, Method, SNRthresh)
            if np.round(float(centroid['y'])) >= 0 and np.round(float(centroid['y'])) <= cents_grid.shape[0]-1 and \
                np.round(float(centroid['x'])) >= 0 and np.round(float(centroid['x'])) <= cents_grid.shape[1]-1:
                cents_grid[(int(np.round(float(centroid['y']))), int(np.round(float(centroid['x']))))] = 1
            cents.append([float(centroid['y']), float(centroid['x'])])
            cent_x.append(float(centroid['x']))
            cent_y.append(float(centroid['y']))
            sigmax.append(centroid['GauSigx'])
            sigmay.append(centroid['GauSigy'])
            offset_list.append(centroid['offset'])
        sigmax_avg = np.mean(sigmax)
        sigmay_avg = np.mean(sigmay)
        offset_avg = np.mean(offset_list)
        points_not_9 = np.delete(points, nine_ind, axis=0)
        for i in points_not_9:
            # row = y[i][1]
            # col = y[i][2]
            row = i[1]
            col = i[0]
            if Gauss_sigma_local:
                centroid = FGCentroid3(pix_corr_x, pix_corr_y, M_in, row, col, box_size, Method, SNRthresh, bright_x, bright_y, sigmax_avg=sigmax_avg, sigmay_avg=sigmay_avg)
                #centroid = FGCentroid2(M_in, row, col, box_size, Method, SNRthresh, sigmax_avg=sigmax_avg, sigmay_avg=sigmay_avg)
            if not Gauss_sigma_local:
                centroid = FGCentroid3(pix_corr_x, pix_corr_y, M_in, row, col, box_size, Method, SNRthresh, bright_x, bright_y)
                #centroid = FGCentroid2(M_in, row, col, box_size, Method, SNRthresh)
            #centroid ={'y': row, 'x': col} This was used just to see how well the peaks themselves did for centroids
            temp_box_size = box_size
            while np.isnan(float(centroid['y'])) or np.isnan(float(centroid['x'])) or centroid['y'] < 0 or centroid['x'] < 0 or centroid['y'] > M_in.shape[0]-1 or centroid['x'] > M_in.shape[1]-1:
                temp_box_size -= 1
                centroid = FGCentroid3(pix_corr_x, pix_corr_y, M_in, row, col, temp_box_size, Method, SNRthresh, bright_x, bright_y)
                #centroid = FGCentroid2(M_in, row, col, temp_box_size, Method, SNRthresh)
            if np.round(float(centroid['y'])) >= 0 and np.round(float(centroid['y'])) <= cents_grid.shape[0]-1 and \
                np.round(float(centroid['x'])) >= 0 and np.round(float(centroid['x'])) <= cents_grid.shape[1]-1:
                cents_grid[(int(np.round(float(centroid['y']))), int(np.round(float(centroid['x']))))] = 1
            cents.append([float(centroid['y']), float(centroid['x'])])
            cent_x.append(float(centroid['x']))
            cent_y.append(float(centroid['y']))
            pass
        cent_x = np.array(cent_x)
        cent_y = np.array(cent_y)

    if Method == 'FFT': #XXX not yet equipped for pix pos corrections
        # these are ordered from brightest to dimmest; i=0 for brightest, i=1 for next brightest, which has best chance for good FastGaussian centroiding
        # brightest:
        br_row = y[0][1]
        br_col = y[0][2]
        br_cent = FGCentroid2(M_in, br_row, br_col, box_size, 'Gaussian', SNRthresh)
        if np.round(float(br_cent['y'])) >= 0 and np.round(float(br_cent['y'])) <= cents_grid.shape[0]-1 and \
            np.round(float(br_cent['x'])) >= 0 and np.round(float(br_cent['x'])) <= cents_grid.shape[1]-1:
            cents_grid[(int(np.round(float(br_cent['y']))), int(np.round(float(br_cent['x']))))] = 1
        cents.append([float(br_cent['y']), float(br_cent['x'])])
        cent_x.append(float(br_cent['x']))
        cent_y.append(float(br_cent['y']))
        # next brightest should be next to brightest in the frame
        first_row = y[1][1]
        first_col = y[1][2]
        first_cent = FGCentroid2(M_in, first_row, first_col, box_size, 'Gaussian', SNRthresh)
        if np.round(float(first_cent['y'])) >= 0 and np.round(float(first_cent['y'])) <= cents_grid.shape[0]-1 and \
            np.round(float(first_cent['x'])) >= 0 and np.round(float(first_cent['x'])) <= cents_grid.shape[1]-1:
            cents_grid[(int(np.round(float(first_cent['y']))), int(np.round(float(first_cent['x']))))] = 1
        cents.append([float(first_cent['y']), float(first_cent['x'])])
        cent_x.append(float(first_cent['x']))
        cent_y.append(float(first_cent['y']))

        # get array of x and y coords for remaining cents
        pks_row = np.array([y[j][1] for j in range(2, len(y))])
        pks_col = np.array([y[j][2] for j in range(2, len(y))])
        # pks_row = np.append(y[0][1], pks_row)
        # pks_col = np.append(y[0][2], pks_col)
        cent_row = first_cent['y']
        cent_col = first_cent['x']
        while len(pks_row) != 0:
            print('centroids left:', len(pks_row))
            #next_ind = np.argmin(((pks_row-first_row)**2+(pks_col-first_col)**2)**0.5)
            next_ind = np.argmin(((pks_row-cent_row)**2+(pks_col-cent_col)**2)**0.5)
            next_row = pks_row[next_ind]
            next_col = pks_col[next_ind]
            next_cent = FFTCorrelate2(M_in, first_row, first_col, cent_row, cent_col, next_row, next_col, box_size)
            if np.round(float(next_cent['y'])) >= 0 and np.round(float(next_cent['y'])) <= cents_grid.shape[0]-1 and \
                np.round(float(next_cent['x'])) >= 0 and np.round(float(next_cent['x'])) <= cents_grid.shape[1]-1:
                cents_grid[(int(np.round(float(next_cent['y']))), int(np.round(float(next_cent['x']))))] = 1
            cents.append([float(next_cent['y']), float(next_cent['x'])])
            cent_x.append(float(next_cent['x']))
            cent_y.append(float(next_cent['y']))
            cent_row = next_cent['y']
            cent_col = next_cent['x']
            first_row = next_row
            first_col = next_col
            pks_row = np.delete(pks_row, next_ind)
            pks_col = np.delete(pks_col, next_ind)
        cent_x = np.array(cent_x)
        cent_y = np.array(cent_y)

    if Method == 'FFT2': #XXX not yet equipped for pix pos corrections
        # these are ordered from brightest to dimmest; i=0 for brightest, i=1 for next brightest, which has best chance for good FastGaussian centroiding
        # brightest:
        br_row = y[0][1]
        br_col = y[0][2]
        br_cent = FGCentroid2(M_in, br_row, br_col, box_size, 'Gaussian', SNRthresh)
        if np.round(float(br_cent['y'])) >= 0 and np.round(float(br_cent['y'])) <= cents_grid.shape[0]-1 and \
            np.round(float(br_cent['x'])) >= 0 and np.round(float(br_cent['x'])) <= cents_grid.shape[1]-1:
            cents_grid[(int(np.round(float(br_cent['y']))), int(np.round(float(br_cent['x']))))] = 1
        cents.append([float(br_cent['y']), float(br_cent['x'])])
        cent_x.append(float(br_cent['x']))
        cent_y.append(float(br_cent['y']))
        # next brightest should be next to brightest in the frame
        first_row = y[1][1]
        first_col = y[1][2]
        first_cent = FGCentroid2(M_in, first_row, first_col, box_size, 'Gaussian', SNRthresh)
        if np.round(float(first_cent['y'])) >= 0 and np.round(float(first_cent['y'])) <= cents_grid.shape[0]-1 and \
            np.round(float(first_cent['x'])) >= 0 and np.round(float(first_cent['x'])) <= cents_grid.shape[1]-1:
            cents_grid[(int(np.round(float(first_cent['y']))), int(np.round(float(first_cent['x']))))] = 1
        cents.append([float(first_cent['y']), float(first_cent['x'])])
        cent_x.append(float(first_cent['x']))
        cent_y.append(float(first_cent['y']))

        # get array of x and y coords for remaining cents
        pks_row = np.array([y[j][1] for j in range(2, len(y))])
        pks_col = np.array([y[j][2] for j in range(2, len(y))])
        # pks_row = np.append(y[0][1], pks_row)
        # pks_col = np.append(y[0][2], pks_col)
        cent_row = first_cent['y']
        cent_col = first_cent['x']
        for ind in range(len(pks_row)):
            #next_cent = FFTCorrelate2(M_in, cent_row, cent_col, pks_row[ind], pks_col[ind], box_size)
            next_cent = FFTCorrelate2(M_in, first_row, first_col, cent_row, cent_col, pks_row[ind], pks_col[ind], box_size)
            if np.round(float(next_cent['y'])) >= 0 and np.round(float(next_cent['y'])) <= cents_grid.shape[0]-1 and \
                np.round(float(next_cent['x'])) >= 0 and np.round(float(next_cent['x'])) <= cents_grid.shape[1]-1:
                cents_grid[(int(np.round(float(next_cent['y']))), int(np.round(float(next_cent['x']))))] = 1
            cents.append([float(next_cent['y']), float(next_cent['x'])])
            cent_x.append(float(next_cent['x']))
            cent_y.append(float(next_cent['y']))

        cent_x = np.array(cent_x)
        cent_y = np.array(cent_y)

    return cents_grid, cents, cent_x, cent_y


def FFTCorrelate3(M, pkRow, pkCol, nextRow, nextCol, Ncentr):

    nr = len(M)
    nc = len(M[0])
    # first, centroid
    rowMin = max(0, int(np.floor(pkRow)) - math.floor(Ncentr/2))
    rowMax = min(nr-1, int(np.floor(pkRow)) + math.floor(Ncentr/2))
    colMin = max(0, int(np.floor(pkCol)) - math.floor(Ncentr/2))
    colMax = min(nc-1, int(np.floor(pkCol)) + math.floor(Ncentr/2))
    # rowInds = list(np.arange(rowMin, rowMax+1))
    # colInds = list(np.arange(colMin, colMax+1))
    # next peak
    n_rowMin = max(0, nextRow - math.floor(Ncentr/2))
    n_rowMax = min(nr-1, nextRow + math.floor(
        Ncentr/2))
    n_colMin = max(0, nextCol - math.floor(Ncentr/2))
    n_colMax = min(nc-1, nextCol + math.floor(Ncentr/2))
    # n_rowInds = list(np.arange(n_rowMin, n_rowMax+1))
    # n_colInds = list(np.arange(n_colMin, n_colMax+1))

    pk_image = M[rowMin:rowMax+1, colMin:colMax+1]
    n_image = M[n_rowMin:n_rowMax+1, n_colMin:n_colMax+1]

    # pad each image with zeros to avoid edge effects; both images are square
    # and of same shape by design
    pad = pk_image.shape[0]
    pk_image_pad = np.pad(pk_image, pad)
    n_image_pad = np.pad(n_image, pad)

    pk_ft_conj = np.conjugate(np.fft.fft2(np.fft.ifftshift(pk_image_pad)))
    n_ft_conj = np.conjugate(np.fft.fft2(np.fft.ifftshift(n_image_pad)))
    n_ft = np.fft.fft2(np.fft.ifftshift(n_image_pad))
    correlation = np.fft.fftshift(np.fft.ifft2(pk_ft_conj*n_ft))
    x_coords = np.arange(pk_image.shape[1])
    y_coords = np.arange(pk_image.shape[0])
    #offset, A, x0, y0, sx, sy, corr_val = gauss_cross_corr_fit(np.real(correlation))
    A, x0, y0, sx, sy, corr_val = gauss_cross_corr_fit2(np.real(correlation))
    centroid = {}
    # centroid['y'] = pkRow - rowMin + n_rowMin + (res.x[1] - np.median(y_coords)) #res.x[1] - pad + n_rowMin
    # centroid['x'] = pkCol - colMin + n_colMin + (res.x[0] - np.median(x_coords)) #res.x[0] - pad + n_colMin
    centroid['y'] = (y0 - pad + n_rowMin) + ((pkRow - rowMin) - np.median(y_coords))
    centroid['x'] = (x0 - pad + n_colMin) + ((pkCol - colMin) - np.median(x_coords))

    return centroid


def FFTCorrelate2(M, pkRow, pkCol, firstcent_row, firstcent_col, nextRow, nextCol, Ncentr):

    nr = len(M)
    nc = len(M[0])
    # first, centroid
    rowMin = max(0, int(np.floor(pkRow)) - math.floor(Ncentr/2))
    rowMax = min(nr-1, int(np.floor(pkRow)) + math.floor(Ncentr/2))
    colMin = max(0, int(np.floor(pkCol)) - math.floor(Ncentr/2))
    colMax = min(nc-1, int(np.floor(pkCol)) + math.floor(Ncentr/2))
    # rowInds = list(np.arange(rowMin, rowMax+1))
    # colInds = list(np.arange(colMin, colMax+1))
    # next peak
    n_rowMin = max(0, nextRow - math.floor(Ncentr/2))
    n_rowMax = min(nr-1, nextRow + math.floor(Ncentr/2))
    n_colMin = max(0, nextCol - math.floor(Ncentr/2))
    n_colMax = min(nc-1, nextCol + math.floor(Ncentr/2))
    # n_rowInds = list(np.arange(n_rowMin, n_rowMax+1))
    # n_colInds = list(np.arange(n_colMin, n_colMax+1))

    pk_image = M[rowMin:rowMax+1, colMin:colMax+1]
    n_image = M[n_rowMin:n_rowMax+1, n_colMin:n_colMax+1]

    # pad each image with zeros to avoid edge effects; both images are square
    # and of same shape by design
    pad = pk_image.shape[0]
    pk_image_pad = np.pad(pk_image, pad)
    n_image_pad = np.pad(n_image, pad)

    pk_ft_conj = np.conjugate(np.fft.fft2(np.fft.ifftshift(pk_image_pad)))
    n_ft_conj = np.conjugate(np.fft.fft2(np.fft.ifftshift(n_image_pad)))
    n_ft = np.fft.fft2(np.fft.ifftshift(n_image_pad))
    correlation = np.fft.fftshift(np.fft.ifft2(pk_ft_conj*n_ft))

    # interpolate and find local max
    x_coords = np.arange(pk_image_pad.shape[1])
    y_coords = np.arange(pk_image_pad.shape[0])
    f = RectBivariateSpline(x_coords, y_coords, np.abs(correlation), kx=5, ky=5)#, s=0)
    #f = SmoothBivariateSpline(x_coords, y_coords, np.real(correlation), kx=5, ky=5)
    #f = LSQBivariateSpline(x_coords, y_coords, np.real(correlation), kx=5, ky=5)#, s=0)
    df = f.partial_derivative(1, 1)
    def func(xy):
        x = xy[1] #xy[0]
        y = xy[0] #xy[1]
        return -f(x, y)
    x_orig = np.arange(pk_image.shape[1])
    y_orig = np.arange(pk_image.shape[0])
    xn = np.arange(n_image.shape[1])
    yn = np.arange(n_image.shape[0])
    # lb = [pad, pad]
    # ub = [pk_image_pad.shape[1] - pad, pk_image_pad.shape[0] - pad]
    # lb = [pk_image_pad.shape[1]/2-5, pk_image_pad.shape[0]/2-5]
    # ub = [pk_image_pad.shape[1]/2+5, pk_image_pad.shape[0]/2+5]
    # lb = [pad + pkCol-colMin -5, pad + pkRow-rowMin -5]
    # ub = [pad + pkCol-colMin +5, pad + pkRow-rowMin +5]
    lb = [0, 0]
    ub = [x_coords.max(), y_coords.max()]
    bounds = Bounds(lb=np.array(lb),
                                ub=np.array(ub))
    res = minimize(func, x0=np.array([np.median(x_coords), np.median(y_coords)]), bounds=bounds, tol=1e-30)

    centroid = {}
    # centroid['y'] = pkRow - rowMin + n_rowMin + (res.x[1] - np.median(y_coords)) #res.x[1] - pad + n_rowMin
    # centroid['x'] = pkCol - colMin + n_colMin + (res.x[0] - np.median(x_coords)) #res.x[0] - pad + n_colMin
    centroid['y'] = firstcent_row - rowMin + n_rowMin + res.x[1] - np.median(y_coords)
    centroid['x'] = firstcent_col - colMin + n_colMin + res.x[0] - np.median(x_coords)
    if False:
        CIS_truth_x = np.load(r'Pat_simulated_centroids/Gauss_noise_shift_truth_x.npy')
        CIS_truth_y = np.load(r'Pat_simulated_centroids/Gauss_noise_shift_truth_y.npy')
        min_ind = np.argmin((CIS_truth_x - centroid['x'])**2 + (CIS_truth_y - centroid['y'])**2)
        if (not np.isclose(CIS_truth_x[min_ind], centroid['x'], atol=0.05) or
            not np.isclose(CIS_truth_y[min_ind], centroid['y'], atol=0.05)):
            print('bad centroiding of (x,y) = ', (centroid['x'],centroid['y']), ' vs ', (CIS_truth_x[min_ind], CIS_truth_y[min_ind]))
    return centroid

def FFTCorrelate(M, pkRow, pkCol, firstcent_row, firstcent_col, nextRow, nextCol, Ncentr):

    nr = len(M)
    nc = len(M[0])
    # first, centroid
    rowMin = max(0, int(np.floor(pkRow)) - math.floor(Ncentr/2))
    rowMax = min(nr-1, int(np.floor(pkRow)) + math.floor(Ncentr/2))
    colMin = max(0, int(np.floor(pkCol)) - math.floor(Ncentr/2))
    colMax = min(nc-1, int(np.floor(pkCol)) + math.floor(Ncentr/2))
    # rowInds = list(np.arange(rowMin, rowMax+1))
    # colInds = list(np.arange(colMin, colMax+1))
    # next peak
    n_rowMin = max(0, nextRow - math.floor(Ncentr/2))
    n_rowMax = min(nr-1, nextRow + math.floor(Ncentr/2))
    n_colMin = max(0, nextCol - math.floor(Ncentr/2))
    n_colMax = min(nc-1, nextCol + math.floor(Ncentr/2))
    # n_rowInds = list(np.arange(n_rowMin, n_rowMax+1))
    # n_colInds = list(np.arange(n_colMin, n_colMax+1))

    pk_image = M[rowMin:rowMax+1, colMin:colMax+1]
    n_image = M[n_rowMin:n_rowMax+1, n_colMin:n_colMax+1]

    # pad each image with zeros to avoid edge effects; both images are square
    # and of same shape by design
    pk_image_pad = np.pad(pk_image, 3*pk_image.shape[0])
    n_image_pad = np.pad(n_image, 3*n_image.shape[0])

    pk_ft_conj = np.conjugate(np.fft.fft2(np.fft.ifftshift(pk_image_pad)))
    n_ft_conj = np.conjugate(np.fft.fft2(np.fft.ifftshift(n_image_pad)))
    #n_ft = np.fft.fft2(np.fft.ifftshift(n_image_pad))
    n_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(n_image_pad)))
    correlation = np.real(np.fft.fftshift(np.fft.ifft2(pk_ft_conj*n_ft)))
    # cor_row, cor_col = np.where(correlation == np.max(correlation))

    # fx_coords = np.arange(pk_ft_conj.shape[1])
    # fy_coords = np.arange(pk_ft_conj.shape[0])
    fx_coords = np.fft.fftshift(np.fft.fftfreq(pk_image_pad.shape[1]))
    fy_coords = np.fft.fftshift(np.fft.fftfreq(pk_image_pad.shape[0]))
    fx, fy = np.meshgrid(fx_coords, fy_coords)
    if False:
        #X_mat = 2j*np.pi*np.array((fx.ravel(), fy.ravel())).T
        X_mat = -2j*np.pi*np.array((fx.ravel(), fy.ravel(), np.log(pk_ft_conj).ravel()/(-2j*np.pi))).T
        Y = np.vstack(np.log(n_ft_conj.ravel()))
        #Y = np.vstack(np.log(n_ft_conj.ravel()/pk_ft_conj.ravel()))
        x_shift, y_shift, gamma = np.linalg.pinv(X_mat)@Y
    if False:
        #X_mat = 2j*np.pi*np.array((fx.ravel(), fy.ravel())).T
        X_mat = -2j*np.pi*np.array((fx.ravel(), fy.ravel())).T
        Y = np.vstack(np.log(n_ft_conj.ravel()) - np.log(pk_ft_conj.ravel()))
        #Y = np.vstack(np.log(n_ft_conj.ravel()/pk_ft_conj.ravel()))
        x_shift, y_shift = np.linalg.pinv(X_mat)@Y
    if True:
        #pk_ft = np.fft.fft2(np.fft.ifftshift(pk_image_pad))
        pk_ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pk_image_pad)))
        def correlate(xy, a, b):
            (x, y) = xy
            #return np.real(pk_ft.ravel()*np.e**(-2j*np.pi*(x*a+y*b)))
            return np.append(np.real(pk_ft.ravel()*np.e**(2j*np.pi*(x*a+y*b))),
                             np.imag(pk_ft.ravel()*np.e**(2j*np.pi*(x*a+y*b))))
        XY = np.vstack((fx.ravel(), fy.ravel()))
        init_guess = (2, 2) #(2,2)
        ub = [colMax-colMin, rowMax-rowMin]
        #ub = [2000, 2**16-1, len(data[0]), len(data), 10, 10]
        lb = [-(colMax-colMin), -(rowMax-rowMin)]
        bounds = (lb, ub)
        popt, pcov = curve_fit(correlate, XY, np.append(np.real((n_ft).ravel()),np.imag((n_ft).ravel())), p0=init_guess, bounds=bounds, maxfev=1e6, xtol=1e-15)
        x_shift = [popt[0]]
        y_shift = [popt[1]]
        # val = correlate(XY, x_shift[0], y_shift[0])
        # neg_val = correlate(XY, -x_shift[0], -y_shift[0])
        # val_res = np.sum((val.ravel() - np.real(n_ft.ravel()))**2)
        # neg_val_res = np.sum((neg_val.ravel() - np.real(n_ft.ravel()))**2)

    centroid = {}
    x_orig = np.arange(pk_image.shape[1])
    y_orig = np.arange(pk_image.shape[0])
    xn = np.arange(n_image.shape[1])
    yn = np.arange(n_image.shape[0])
    # take real part of shift since it SHOULD be real
    # centroid['y'] = n_rowMin + (pkRow-rowMin) + np.real(y_shift[0])
    # centroid['x'] = n_colMin + (pkCol-colMin) + np.real(x_shift[0])
    centroid['y'] = n_rowMin + np.median(yn) + firstcent_row-(np.median(y_orig)+rowMin) - np.real(y_shift[0])
    centroid['x'] = n_colMin + np.median(xn) + firstcent_col-(np.median(x_orig)+colMin) - np.real(x_shift[0])

    if False:
        CIS_truth_x = np.load(r'Pat_simulated_centroids/Gauss_noise_shift_truth_x.npy')
        CIS_truth_y = np.load(r'Pat_simulated_centroids/Gauss_noise_shift_truth_y.npy')
        min_ind = np.argmin((CIS_truth_x - centroid['x'])**2 + (CIS_truth_y - centroid['y'])**2)
        if (not np.isclose(CIS_truth_x[min_ind], centroid['x'], atol=0.05) or
            not np.isclose(CIS_truth_y[min_ind], centroid['y'], atol=0.05)):
            print('bad centroiding of (x,y) = ', (centroid['x'],centroid['y']), ' vs ', (CIS_truth_x[min_ind], CIS_truth_y[min_ind]))
    return centroid

def FGCentroid(M, pkRow, pkCol, Ncentr, Method, sizeU, *args):

    nr = len(M)
    nc = len(M[0])
    rowMin = max(0, pkRow - math.floor(Ncentr/2))
    rowMax = min(nr-1, pkRow + math.floor(Ncentr/2))
    colMin = max(0, pkCol - math.floor(Ncentr/2))
    colMax = min(nc-1, pkCol + math.floor(Ncentr/2))
    rowInds = list(np.arange(rowMin, rowMax+1))
    colInds = list(np.arange(colMin, colMax+1))
    # rowInds = [i for i in range(rowMin, rowMax + 1)]
    # colInds = [i for i in range(colMin, colMax + 1)]

    image = M[rowMin:rowMax+1, colMin:colMax+1]
    # image = []
    # for r in range(rowMin, rowMax+1):
    #     temp = []
    #     for c in range(colMin, colMax+1):
    #         temp.append(M[r][c])
    #     image.append(temp)

    x, y = np.meshgrid(colInds, rowInds)
    #N = [len(image[i]) for i in range(len(image))]

    centroid = {}
    if Method == 'FirstMoment':
        Msubtot = sumMat(image)
        xCentroid = sumMat(np.multiply(x, image))
        yCentroid = sumMat(np.multiply(y, image))
        centroid['row'] = yCentroid/Msubtot
        centroid['col'] = xCentroid/Msubtot
    elif Method == 'FastGaussian':
        # based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6163372/
        nSamp = 5 #XXX should try sizeU?  I did before, and the results were still bad.  Look into createMini().
        I, ui = maxk(flatten(image), nSamp)
        # since flatten() takes the transpose first, below is ux, uy
        ux, uy = np.unravel_index(ui, (len(colInds), len(rowInds)))
        # ux = items_from_indices(flatten(x), ui)
        # uy = items_from_indices(flatten(y), ui)
        M1 = createMini(ux, uy, np.array(I))  # M1 = createMini(ux, uy, I)
        solv1 = sympy.Matrix(M1).rref() #maybe ain't the best, but it is working for now

        paramV1 = [item[-1] for item in solv1[0].tolist()]
        [A1, xc1, yc1, sigma1] = Pars2GaussMini(paramV1)
        #up to here should be good
        SigEst1 = calcSigEst(A1, x, xc1, sigma1, y, yc1)
        Noise = np.absolute(np.array(image) - SigEst1)
        #gonna need to make the shape better in this next line potentially
        #SNR = np.minimum(np.divide(SigEst1, Noise), [[8 for _ in range(len(Noise[0]))] for _ in range(len(Noise))])
        SNR = np.minimum(np.divide(SigEst1, Noise), 8*np.ones_like(Noise)) #XXX significance of 8??
        #SNR = np.divide(SigEst1, Noise)
        #you could implement debugmode later

        #threshold = 3

        #sizeU = 5
        values, indices = maxk(flatten(SNR), min(sizeU, len(flatten(SNR))))
        threshold = min(values)

        Umap = makeUMap(SNR, threshold)


        sizeU = sum(flatten(Umap))

        #threshAdj = False

        # while sizeU < 5:
        #     threshold = threshold * 0.99
        #     Umap = makeUMap(SNR, threshold)
        #     sizeU = sum(flatten(Umap))
        #     threshAdj = True


        Urow, Ucol = find_ones(Umap)
        #I2 = makeI2(Umap, image)
        I2 = image[Urow, Ucol]
        # Ux = [x[0][item] for item in Ucol]
        # Uy = [y[item][0] for item in Urow]
        Ux = np.array(Ucol) + colMin
        Uy = np.array(Urow) + rowMin
        M2 = createMini(Ux,Uy,I2)
        solv2 = sympy.Matrix(M2).rref()
        paramV2 = [item[-1] for item in solv2[0].tolist()]
        [A2, xc2, yc2, sigma2] = Pars2GaussMini(paramV2)
        SigEst2 = calcSigEst(A2, x, xc2, sigma2, y, yc2)
        if np.isnan(float(xc2)) or np.isnan(float(yc2)) or float(xc2)<0 or float(yc2)<0:
            print('nan output parameters!') # testing purposes

        #subtracting 1 to bring it back to python coordinates
        centroid['row'] = yc2 - 1
        centroid['col'] = xc2 - 1
        centroid['GauImg'] = SigEst2
        centroid['GauAmp'] = A2
        centroid['GauSig'] = sigma2
        centroid['offset'] = 0 # by assumption with FastGaussian method
    else:
        raise Exception('unsupported centroiding method!')

    if len(args) == 0:
        centroid['y'] = centroid['row']
        centroid['x'] = centroid['col']
    elif len(args) == 2:
                #might be worth checking out this next part.  I haven't tested it yet, so there might be some issues...
        rowAxis = args[0]
        colAxis = args[1]
        rowLo = math.floor(centroid['row'])
        colLo = math.floor(centroid['col'])
        if rowLo < 0 or colLo < 0 or centroid['row'] >= rowMax or centroid['col'] >= colMax:
            raise Exception('This function is only designed to find centroids within the image provided. If the centroid is within the image, then the image may be undersampled.')
        centroid['y'] = rowAxis[rowLo] + (rowAxis[rowLo+1]-rowAxis[rowLo]) * (centroid['row'] - rowLo)
        centroid['x'] = colAxis[colLo] + (colAxis[colLo+1]-colAxis[colLo]) * (centroid['col'] - colLo)
    else:
        raise Exception('wrong number of arguments')


    return centroid

def Airy_1spot(xy, x0, y0, prim_mm, sec_mm, A, B, offset, a, b):
    '''Obscurated Airy disc.  Intended for flattened arrays.'''
    (x, y) = xy
    r = np.sqrt(a*(x-x0)**2 + b*(y-y0)**2)
    Airy = np.zeros_like(r)
    cols = np.where(r!=0)
    ocols = np.where(r==0)
    Airy[cols] = (np.pi*(prim_mm/2)**2*j1(2*np.pi*r[r!=0]*A)/(np.pi*r[r!=0]*A) - np.pi*(sec_mm/2)**2*j1(2*np.pi*r[r!=0]*B)/(np.pi*r[r!=0]*B))**2
    Airy[ocols] = (np.pi*(prim_mm/2)**2 - np.pi*(sec_mm/2)**2)**2
    return Airy.astype(float) + offset

def Airy_arctan_1spot(xy, x0, y0, prim_mm, sec_mm, A, B, offset, a, b, C, D, c, d, x1, y1):
    '''Obscurated Airy disc.  Intended for flattened arrays.'''
    (x, y) = xy
    r = np.sqrt(a*(x-x0)**2 + b*(y-y0)**2)
    Airy = np.zeros_like(r)
    cols = np.where(r!=0)
    ocols = np.where(r==0)
    Airy[cols] = (np.pi*(prim_mm/2)**2*j1(2*np.pi*r[r!=0]*A)/(np.pi*r[r!=0]*A) - np.pi*(sec_mm/2)**2*j1(2*np.pi*r[r!=0]*B)/(np.pi*r[r!=0]*B))**2
    Airy[ocols] = (np.pi*(prim_mm/2)**2 - np.pi*(sec_mm/2)**2)**2
    return Airy.astype(float)*(C*np.arctan(c*(x-x1)+d*(y-y1)) +D) + offset

def Airy_spot(xy, x0, y0, x1, y1, prim_mm, sec_mm, A, B, offset):#, norm):
    '''Obscurated Airy disc.  Intended for flattened arrays.'''
    (x, y) = xy
    r = np.sqrt(((x-x0))**2 + ((y-y0))**2)
    r1 = np.sqrt(((x-x1))**2 + ((y-y1))**2)
    Airy = np.zeros_like(r)
    Airy1 = Airy.copy()
    cols = np.where(r!=0)
    cols1 = np.where(r1!=0)
    ocols = np.where(r==0)
    ocols1 = np.where(r1==0)
    Airy[cols] = np.pi*(prim_mm/2)**2*j1(2*np.pi*r[r!=0]*A)/(np.pi*r[r!=0]*A)
    Airy1[cols1] = - np.pi*(sec_mm/2)**2*j1(2*np.pi*r1[r1!=0]*B)/(np.pi*r1[r1!=0]*B)
    Airy[ocols] = np.pi*(prim_mm/2)**2
    Airy1[ocols1] = - np.pi*(sec_mm/2)**2
    Airy_out = (Airy + Airy1)**2 + offset
    #Airy_out = (Airy)**2 - (Airy1)**2
    return Airy_out.astype(float)

# can't have offset for Fast Gaussian method b/c that would mess up the linear least squares nature of it
def gauss_spot(xy, offset, A, x0, y0, sx, sy):
    (x, y) = xy
    return offset + A*np.e**(-((x-x0)**2/(2*sx**2) + (y-y0)**2/(2*sy**2)))

def rot_gauss_spot(xy, offset, A, x0, y0, sx, sy, theta):
    (x, y) = xy
    return offset + A*np.e**(-(np.cos(theta)*(x-x0)-np.sin(theta)*(y-y0))**2/(2*sx**2) - (np.sin(theta)*(x-x0)+np.cos(theta)*(y-y0))**2/(2*sy**2))

def gauss_spot_arctan(xy, offset, A, x0, y0, sx, sy, B, C, b, c, x1, y1):
    (x, y) = xy
    return offset + A*np.e**(-((x-x0)**2/(2*sx**2) + (y-y0)**2/(2*sy**2)))*(B*np.arctan(b*(x-x1)+c*(y-y1)) +C)

def arctan_spot(xy, B, C, b, c, x1, y1):
    (x, y) = xy
    return B*np.arctan(b*(x-x1)+c*(y-y1)) +C


def gauss_5_spot(xy, offset, A, x0, y0, sx, sy, A2, x02, y02, sx2, sy2, A3, x03, y03, sx3, sy3, A4, x04, y04, sx4, sy4, A5, x05, y05, sx5, sy5):
    (x, y) = xy
    return offset + A*np.e**(-((x-x0)**2/(2*sx**2) + (y-y0)**2/(2*sy**2))) + A2*np.e**(-((x-x02)**2/(2*sx2**2) + (y-y02)**2/(2*sy2**2))) + \
            A3*np.e**(-((x-x03)**2/(2*sx3**2) + (y-y03)**2/(2*sy3**2))) + A4*np.e**(-((x-x04)**2/(2*sx4**2) + (y-y04)**2/(2*sy4**2))) + \
            A5*np.e**(-((x-x05)**2/(2*sx5**2) + (y-y05)**2/(2*sy5**2)))

def arctan_fit_corr(data, x_corr, y_corr):
    XY = np.vstack((x_corr, y_corr))
    init_guess = (4000, 200, 1, 1, np.mean(x_corr), np.mean(y_corr))
    #init_guess = (0, 30, len(data[0])/2, len(data)/2, 2, 2)
    ub = [2**16-1, 2**16-1, 100, 100, x_corr.max(), y_corr.max()]
    #ub = [2000, 2**16-1, len(data[0]), len(data), 10, 10]
    lb = [0, -40000, -100, -100, x_corr.min(), y_corr.min()]
    bounds = (lb, ub)
    popt, pcov = curve_fit(arctan_spot, XY, data, bounds=bounds, p0=init_guess, maxfev=1e5, xtol=1e-15)
    gauss_val = arctan_spot(XY, *popt)
    return *popt, gauss_val

def Airy_1fit_corr(data, x_corr, y_corr):
    XY = np.vstack((x_corr, y_corr))
    init_guess = (np.mean(x_corr), np.mean(y_corr), 6500, 3000, 6500, 3000, 200, 1, 1)
    #init_guess = (0, 30, len(data[0])/2, len(data)/2, 2, 2)
    ub = [2*x_corr.max(), 2*y_corr.max(), 100000, 100000, 100000, 100000, 2**16-1, 100000, 100000] #10000, 10000]
    #ub = [2000, 2**16-1, len(data[0]), len(data), 10, 10]
    lb = [-x_corr.max()+2*x_corr.min(), -y_corr.max()+2*y_corr.min(), 0, 0, 0 ,0, -100000, 0, 0]
    bounds = (lb, ub)
    popt, pcov = curve_fit(Airy_1spot, XY, data, bounds=bounds, p0=init_guess, maxfev=1e5, xtol=1e-15)
    gauss_val = Airy_1spot(XY, *popt)
    return *popt, gauss_val

def Airy_arctan_1fit_corr(data, x_corr, y_corr):
    XY = np.vstack((x_corr, y_corr))
    init_guess = (np.mean(x_corr), np.mean(y_corr), 6500, 3000, 6500, 3000, 200, 1, 1, 1, 1, 1, 1, np.mean(x_corr), np.mean(y_corr))
    #init_guess = (0, 30, len(data[0])/2, len(data)/2, 2, 2)
    ub = [2*x_corr.max(), 2*y_corr.max(), 100000, 100000, 100000, 100000, 2**16-1, 100000, 100000, 1000, 1000, 1000, 1000, x_corr.max(), y_corr.max()] #10000, 10000]
    #ub = [2000, 2**16-1, len(data[0]), len(data), 10, 10]
    lb = [-x_corr.max()+2*x_corr.min(), -y_corr.max()+2*y_corr.min(), 0, 0, 0 ,0, -100000, 0, 0, -1000, -1000, -1000, -1000, x_corr.min(), y_corr.min()]
    bounds = (lb, ub)
    popt, pcov = curve_fit(Airy_arctan_1spot, XY, data, bounds=bounds, p0=init_guess, maxfev=1e5, xtol=1e-15)
    gauss_val = Airy_arctan_1spot(XY, *popt)
    return *popt, gauss_val

def Airy_fit_corr(data, x_corr, y_corr):
    XY = np.vstack((x_corr, y_corr))
    init_guess = (np.mean(x_corr), np.mean(y_corr), np.mean(x_corr), np.mean(y_corr), 6500, 3000, 6500, 3000, 200)#, 100)
    #init_guess = (0, 30, len(data[0])/2, len(data)/2, 2, 2)
    ub = [x_corr.max(), y_corr.max(), 2*x_corr.max(), 2*y_corr.max(), 100000, 10000, 10000, 10000, 2**16-1]#, 10000]
    #ub = [2000, 2**16-1, len(data[0]), len(data), 10, 10]
    lb = [x_corr.min(), y_corr.min(), -x_corr.max(), -y_corr.max(), 0, 0, 0, 0, -100000]#, 0]
    bounds = (lb, ub)
    popt, pcov = curve_fit(Airy_spot, XY, data, bounds=bounds, p0=init_guess, maxfev=1e5, xtol=1e-15)
    gauss_val = Airy_spot(XY, *popt)
    return *popt, gauss_val

def gauss_5_fit_corr(data, x_corr, y_corr):
    XY = np.vstack((x_corr, y_corr))
    init_guess = tuple([200]+ [5000, np.mean(x_corr), np.mean(y_corr), 2, 2] +
                       [5000, np.mean(x_corr)+5, np.mean(y_corr), 2, 2] +
                        [5000, np.mean(x_corr)-5, np.mean(y_corr), 2, 2] +
                        [5000, np.mean(x_corr), np.mean(y_corr) + 5, 2, 2] +
                        [5000, np.mean(x_corr), np.mean(y_corr) - 5, 2, 2])
    #init_guess = (0, 30, len(data[0])/2, len(data)/2, 2, 2)
    ub = [2**16-1] +5*[2**16-1, x_corr.max(), y_corr.max(), 10, 10] #10000, 10000]
    #ub = [2000, 2**16-1, len(data[0]), len(data), 10, 10]
    lb = [0]+5*[0, x_corr.min(), y_corr.min(), 0, 0]
    bounds = (lb, ub)
    popt, pcov = curve_fit(gauss_5_spot, XY, data, bounds=bounds, p0=init_guess, maxfev=1e5, xtol=1e-15)
    #gauss_val = gauss_5_spot(XY, *popt)
    return popt #, gauss_val

def gauss_fit_corr(data, x_corr, y_corr):
    XY = np.vstack((x_corr, y_corr))
    init_guess = (200, 5000, np.mean(x_corr), np.mean(y_corr), 2, 2)
    #init_guess = (0, 30, len(data[0])/2, len(data)/2, 2, 2)
    ub = [2**16-1, 2**16-1, x_corr.max(), y_corr.max(), 10, 10] #10000, 10000]
    #ub = [2000, 2**16-1, len(data[0]), len(data), 10, 10]
    lb = [0, 0, x_corr.min(), y_corr.min(), 0, 0]
    bounds = (lb, ub)
    popt, pcov = curve_fit(gauss_spot, XY, data, bounds=bounds, p0=init_guess, maxfev=1e5, xtol=1e-15)
    gauss_val = gauss_spot(XY, *popt)
    return *popt, gauss_val

def rot_gauss_fit_corr(data, x_corr, y_corr):
    XY = np.vstack((x_corr, y_corr))
    init_guess = (200, 5000, np.mean(x_corr), np.mean(y_corr), 2, 2, 0)
    #init_guess = (0, 30, len(data[0])/2, len(data)/2, 2, 2)
    ub = [2**16-1, 2**16-1, x_corr.max(), y_corr.max(), 10, 10, np.pi] #10000, 10000]
    #ub = [2000, 2**16-1, len(data[0]), len(data), 10, 10]
    lb = [0, 0, x_corr.min(), y_corr.min(), 0, 0, 0]
    bounds = (lb, ub)
    popt, pcov = curve_fit(rot_gauss_spot, XY, data, bounds=bounds, p0=init_guess, maxfev=1e5, xtol=1e-15)
    gauss_val = rot_gauss_spot(XY, *popt)
    return *popt, gauss_val

def gauss_arctan_fit_corr(data, x_corr, y_corr):
    XY = np.vstack((x_corr, y_corr))
    #offset, A, x0, y0, sx, sy, B, C, b, c, x1, y1
    init_guess = (200, 5000, np.mean(x_corr), np.mean(y_corr), 2, 2, 1, 1, 1, 1, np.mean(x_corr), np.mean(y_corr))
    #init_guess = (0, 30, len(data[0])/2, len(data)/2, 2, 2)
    ub = [2**16-1, 2**16-1, x_corr.max(), y_corr.max(), 10, 10, 1000, 1000, 1000, 1000, x_corr.max(), y_corr.max()] #10000, 10000]
    #ub = [2000, 2**16-1, len(data[0]), len(data), 10, 10]
    lb = [0, 0, x_corr.min(), y_corr.min(), 0, 0, -1000, -1000, -1000, -1000, x_corr.min(), y_corr.min()]
    bounds = (lb, ub)
    popt, pcov = curve_fit(gauss_spot_arctan, XY, data, bounds=bounds, p0=init_guess, maxfev=1e5, xtol=1e-15)
    gauss_val = gauss_spot_arctan(XY, *popt)
    return *popt, gauss_val

def gauss_fit(data):
    Y = np.arange(0,len(data))
    X = np.arange(0,len(data[0]))
    # central_pixel_x = np.round(np.median(X))
    # central_pixel_y = np.round(np.median(Y))
    X, Y = np.meshgrid(X,Y)
    XY = np.vstack((X.ravel(), Y.ravel()))
    init_guess = (200, 5000, len(data[0])/2, len(data)/2, 2, 2)
    #init_guess = (0, 30, len(data[0])/2, len(data)/2, 2, 2)
    ub = [2**16-1, 2**16-1, len(data[0]), len(data), 10, 10] #10000, 10000]
    #ub = [2000, 2**16-1, len(data[0]), len(data), 10, 10]
    lb = [0, 0, 0, 0, 0, 0]
    bounds = (lb, ub)
    popt, pcov = curve_fit(gauss_spot, XY, data.ravel(), bounds=bounds, p0=init_guess, maxfev=1e5, xtol=1e-15)
    gauss_val = gauss_spot(XY, *popt)
    return *popt, gauss_val

def rot_gauss_fit_fixed_sigma_corr(data, x_corr, y_corr, sx, sy):
    XY = np.vstack((x_corr, y_corr))
    def rot_gauss_spot_fixed_sigma(xy, offset, A, x0, y0, theta):
        (x, y) = xy
        return offset + A*np.e**(-(np.cos(theta)*(x-x0)-np.sin(theta)*(y-y0))**2/(2*sx**2) - (np.sin(theta)*(x-x0)+np.cos(theta)*(y-y0))**2/(2*sy**2))
    init_guess = (200, 5000, np.mean(x_corr), np.mean(y_corr), 0)
    #init_guess = (0, 30, len(data[0])/2, len(data)/2, 2, 2)
    ub = [2**16-1, 2**16-1, x_corr.max(), y_corr.max(), np.pi] #10000, 10000]
    #ub = [2000, 2**16-1, len(data[0]), len(data), 10, 10]
    lb = [0, 0, x_corr.min(), y_corr.min(), 0]
    bounds = (lb, ub)
    popt, pcov = curve_fit(rot_gauss_spot_fixed_sigma, XY, data, bounds=bounds, p0=init_guess, maxfev=1e5, xtol=1e-15)
    gauss_val = rot_gauss_spot_fixed_sigma(XY, *popt)
    return *popt, gauss_val

def gauss_fit_fixed_sigma_corr(data, x_corr, y_corr, sx, sy):
    XY = np.vstack((x_corr, y_corr))
    def gauss_spot_fixed_sigma(xy, offset, A, x0, y0):
        (x, y) = xy
        return offset + A*np.e**(-((x-x0)**2/(2*sx**2) + (y-y0)**2/(2*sy**2)))
    init_guess = (200, 5000, np.mean(x_corr), np.mean(y_corr))
    # doesn't really make a difference which init_guess
    #init_guess = (20, 16000, len(data[0])/2, len(data)/2)
    ub = [2**16-1, 2**16-1, x_corr.max(), y_corr.max()]
    lb = [0, 0, x_corr.min(), y_corr.min()]
    # init_guess = (5000, len(data[0])/2, len(data)/2)
    # ub = [2**16-1, len(data[0]), len(data)]
    # lb = [1, 0, 0]
    bounds = (lb, ub)
    popt, pcov = curve_fit(gauss_spot_fixed_sigma, XY, data, bounds=bounds, p0=init_guess, maxfev=1e5, xtol=1e-15)
    gauss_val = gauss_spot_fixed_sigma(XY, *popt)
    return *popt, gauss_val

def gauss_fit_diff_fixed_sigma_corr(data, x_corr, y_corr, sx, sy):
    XY = np.vstack((x_corr, y_corr))
    def gauss_spot_diff_fixed_sigma(xy, offset, A, x0, y0, B, x1, y1, sx1, sy1):
        (x, y) = xy
        return offset + A*np.e**(-((x-x0)**2/(2*sx**2) + (y-y0)**2/(2*sy**2))) -\
            B*np.e**(-((x-x1)**2/(2*sx1**2) + (y-y1)**2/(2*sy1**2)))
    init_guess = (200, 5000, np.mean(x_corr), np.mean(y_corr), 2000, np.mean(x_corr), np.mean(y_corr), 2, 2)
    # doesn't really make a difference which init_guess
    #init_guess = (20, 16000, len(data[0])/2, len(data)/2)
    ub = [2**16-1, 2**16-1, x_corr.max(), y_corr.max(), 2**16-1, 3*x_corr.max(), 3*y_corr.max(), 10, 10]
    lb = [0, 0, x_corr.min(), y_corr.min(), 0, -3*x_corr.min(), -3*y_corr.min(), 0, 0]
    # init_guess = (5000, len(data[0])/2, len(data)/2)
    # ub = [2**16-1, len(data[0]), len(data)]
    # lb = [1, 0, 0]
    bounds = (lb, ub)
    popt, pcov = curve_fit(gauss_spot_diff_fixed_sigma, XY, data, bounds=bounds, p0=init_guess, maxfev=1e5, xtol=1e-15)
    gauss_val = gauss_spot_diff_fixed_sigma(XY, *popt)
    return *popt, gauss_val

def gauss_fit_fixed_sigma(data, sx, sy):
    Y = np.arange(0,len(data))
    X = np.arange(0,len(data[0]))
    # central_pixel_x = np.round(np.median(X))
    # central_pixel_y = np.round(np.median(Y))
    #def gauss_spot_fixed_sigma(xy, A, x0, y0):
    def gauss_spot_fixed_sigma(xy, offset, A, x0, y0):
        (x, y) = xy
        return offset + A*np.e**(-((x-x0)**2/(2*sx**2) + (y-y0)**2/(2*sy**2)))
    X, Y = np.meshgrid(X,Y)
    XY = np.vstack((X.ravel(), Y.ravel()))
    init_guess = (200, 5000, len(data[0])/2, len(data)/2)
    # doesn't really make a difference which init_guess
    #init_guess = (20, 16000, len(data[0])/2, len(data)/2)
    ub = [2**16-1, 2**16-1, len(data[0]), len(data)]
    lb = [0, 0, 0, 0]
    # init_guess = (5000, len(data[0])/2, len(data)/2)
    # ub = [2**16-1, len(data[0]), len(data)]
    # lb = [1, 0, 0]
    bounds = (lb, ub)
    popt, pcov = curve_fit(gauss_spot_fixed_sigma, XY, data.ravel(), bounds=bounds, p0=init_guess, maxfev=1e5, xtol=1e-15)
    gauss_val = gauss_spot_fixed_sigma(XY, *popt)
    return *popt, gauss_val

def sincs_fit(data, x_corr, y_corr):
    XY = np.vstack((x_corr, y_corr))
    def sincs_spot(xy, offset, A, x0, y0, a, b):
        (x, y) = xy
        #A0 + A Sinc^2 (a (x - x0) + b (y - y0)) Sinc^2 (-b (x - x0) +
        #a (y - y0))  See Mathematica file for details
        return offset + A*(np.sinc(a*(x-x0) + b*(y-y0)))**2*(np.sinc(-b*(x-x0) + a*(y-y0)))**2

    init_guess = (200, 5000, np.mean(x_corr), np.mean(y_corr), 1, 1)
    #init_guess = (0, 30, len(data[0])/2, len(data)/2, 2, 2)
    ub = [2**16-1, 2**16-1, x_corr.max(), y_corr.max(), 100, 100] #10000, 10000]
    #ub = [2000, 2**16-1, len(data[0]), len(data), 10, 10]
    lb = [0, 0, x_corr.min(), y_corr.min(), 0, 0]
    bounds = (lb, ub)
    popt, pcov = curve_fit(sincs_spot, XY, data, bounds=bounds, p0=init_guess, maxfev=1e5, xtol=1e-15)
    sincs_val = sincs_spot(XY, *popt)
    return *popt, sincs_val

def sincs_fit2(data, x_corr, y_corr):
    XY = np.vstack((x_corr, y_corr))
    def sincs_spot(xy, offset, A, x0, y0, a, b, c, d):
        (x, y) = xy
        #A0 + A Sinc^2 (a (x - x0) + b (y - y0)) Sinc^2 (-b (x - x0) +
        #a (y - y0))  See Mathematica file for details
        return offset + A*(np.sinc(a*(x-x0) + b*(y-y0)+c))**2*(np.sinc(-b*(x-x0) + a*(y-y0)+d))**2
    init_guess = (200, 5000, np.mean(x_corr), np.mean(y_corr), 1, 1, 0, 0)
    #init_guess = (0, 30, len(data[0])/2, len(data)/2, 2, 2)
    ub = [2**16-1, 2**16-1, x_corr.max(), y_corr.max(), 20, 20, 20, 20] #10000, 10000]
    #ub = [2**16-1, 2**16-1, x_corr.max(), y_corr.max(), 10, 10, 10, 10]
    #ub = [2000, 2**16-1, len(data[0]), len(data), 10, 10]
    lb = [0, 0, x_corr.min(), y_corr.min(), -20, -20, -20, -20]
    #lb = [0, 0, x_corr.min(), y_corr.min(), -10, -10, -10, -10]
    bounds = (lb, ub)
    popt, pcov = curve_fit(sincs_spot, XY, data, bounds=bounds, p0=init_guess, maxfev=1e5, xtol=1e-15)
    sincs_val = sincs_spot(XY, *popt)
    return *popt, sincs_val

def sincs_fit3(data, x_corr, y_corr):
    XY = np.vstack((x_corr, y_corr))
    def sincs_spot(xy, offset, A, x0, y0, a, b, c, d, e, f):
        (x, y) = xy
        #A0 + A Sinc^2 (a (x - x0) + b (y - y0)) Sinc^2 (-b (x - x0) +
        #a (y - y0))  See Mathematica file for details
        return offset + A*(np.sinc(a*(x-x0) + b*(y-y0)+c))**2*(np.sinc(e*(x-x0) + f*(y-y0)+d))**2
    init_guess = (200, 5000, np.mean(x_corr), np.mean(y_corr), 1, 1, 0, 0, 1, 1)
    #init_guess = (0, 30, len(data[0])/2, len(data)/2, 2, 2)
    ub = [2**16-1, 2**16-1, x_corr.max(), y_corr.max(), 1000, 1000, 1000, 1000, 1000, 1000] #10000, 10000]
    #ub = [2**16-1, 2**16-1, x_corr.max(), y_corr.max(), 10, 10, 10, 10]
    #ub = [2000, 2**16-1, len(data[0]), len(data), 10, 10]
    lb = [0, 0, x_corr.min(), y_corr.min(), -1000, -1000, -1000, -1000, -1000, -1000]
    #lb = [0, 0, x_corr.min(), y_corr.min(), -10, -10, -10, -10]
    bounds = (lb, ub)
    popt, pcov = curve_fit(sincs_spot, XY, data, bounds=bounds, p0=init_guess, maxfev=1e5, xtol=1e-15)
    sincs_val = sincs_spot(XY, *popt)
    return *popt, sincs_val

def sincs_fit4(data, x_corr, y_corr):
    XY = np.vstack((x_corr, y_corr))
    def sincs_spot(xy, offset, A, B, x0, y0, a, b, c, d):
        (x, y) = xy
        #A0 + A Sinc^2 (a (x - x0) + b (y - y0)) Sinc^2 (-b (x - x0) +
        #a (y - y0))  See Mathematica file for details
        return offset + (A*np.sinc(a*(x-x0) + b*(y-y0))*(np.sinc(-b*(x-x0) + a*(y-y0))) - B*np.sinc(c*(x-x0) + d*(y-y0))*(np.sinc(-d*(x-x0) + c*(y-y0))))**2
    init_guess = (200, 5000, 5000, np.mean(x_corr), np.mean(y_corr), 1, 1, 1, 1)
    #init_guess = (0, 30, len(data[0])/2, len(data)/2, 2, 2)
    ub = [2**16-1, 2**16-1, 2**16-1, x_corr.max(), y_corr.max(), 1000, 1000, 1000, 1000] #10000, 10000]
    #ub = [2**16-1, 2**16-1, x_corr.max(), y_corr.max(), 10, 10, 10, 10]
    #ub = [2000, 2**16-1, len(data[0]), len(data), 10, 10]
    lb = [0, 0, 0, x_corr.min(), y_corr.min(), -1000, -1000, -1000, -1000]
    #lb = [0, 0, x_corr.min(), y_corr.min(), -10, -10, -10, -10]
    bounds = (lb, ub)
    popt, pcov = curve_fit(sincs_spot, XY, data, bounds=bounds, p0=init_guess, maxfev=1e5, xtol=1e-15)
    sincs_val = sincs_spot(XY, *popt)
    return *popt, sincs_val

def sincs_5_spot(xy, offset, c, d, A, x0, y0, a, A2, x02, y02, a2, A3, x03, y03, a3, A4, x04, y04, a4, A5, x05, y05, a5):
    (x, y) = xy
    X = c*(x-x0) + d*(y-y0)#c*(x) + d*(y) #c*(x-x0) + d*(y-y0)
    Y = -d*(x-x0) + c*(y-y0)#-d*(x) + c*(y) #-d*(x-x0) + c*(y-y0)
    r = np.sqrt((X-x0)**2+(Y-y0)**2)
    r2 = np.sqrt((X-x02)**2+(Y-y02)**2)
    r3 = np.sqrt((X-x03)**2+(Y-y03)**2)
    r4 = np.sqrt((X-x04)**2+(Y-y04)**2)
    r5 = np.sqrt((X-x05)**2+(Y-y05)**2)

    #return offset + A*np.sinc(a*r)**2 + A2*np.sinc(a2*r2)**2 + A3*np.sinc(a3*r3)**2 + A4*np.sinc(a4*r4)**2 + A5*np.sinc(a5*r5)**2
    return offset + A*np.sinc(a*r) + A2*np.sinc(a2*r2) + A3*np.sinc(a3*r3) + A4*np.sinc(a4*r4) + A5*np.sinc(a5*r5)

def sincs_5_fit_corr(data, x_corr, y_corr):
    XY = np.vstack((x_corr, y_corr))
    init_guess = tuple([200, 1, 1]+ [5000, np.mean(x_corr), np.mean(y_corr), 1] +
                       [5000, np.mean(x_corr)+5, np.mean(y_corr), 1] +
                        [5000, np.mean(x_corr)-5, np.mean(y_corr), 1] +
                        [5000, np.mean(x_corr), np.mean(y_corr) + 5, 1] +
                        [5000, np.mean(x_corr), np.mean(y_corr) - 5, 1])
    #init_guess = (0, 30, len(data[0])/2, len(data)/2, 2, 2)
    ub = [2**16-1, 1000, 1000] +5*[2**16-1, x_corr.max(), y_corr.max(), 1000] #10000, 10000]
    #ub = [2000, 2**16-1, len(data[0]), len(data), 10, 10]
    lb = [0, 0, 0]+5*[0, x_corr.min(), y_corr.min(), 0]
    bounds = (lb, ub)
    popt, pcov = curve_fit(sincs_5_spot, XY, data, bounds=bounds, p0=init_guess, maxfev=1e5, xtol=1e-15)
    #gauss_val = gauss_5_spot(XY, *popt)
    return popt #, gauss_val

#######################  2 functions below:  For FFT centroiding

def gauss_cross_corr_fit(data):
    Y = np.arange(0,len(data))
    X = np.arange(0,len(data[0]))
    # central_pixel_x = np.round(np.median(X))
    # central_pixel_y = np.round(np.median(Y))
    #def gauss_spot_fixed_sigma(xy, A, x0, y0):
    def gauss_cross_corr(xy, offset, A, x0, y0, sx, sy):
        (x, y) = xy
        return offset + 2*A*np.e**(-(x-x0)**2/(2*sx**2) - (y-y0)**2/(2*sy**2))
    X, Y = np.meshgrid(X,Y)
    XY = np.vstack((X.ravel(), Y.ravel()))
    init_guess = (500, 2000, len(data[0])/2, len(data)/2, 2, 2)
    ub = [2**16-1, 2**16-1, len(data[0]), len(data), 10000, 10000]
    lb = [0, 0, 0, 0, 0, 0]
    # init_guess = (5000, len(data[0])/2, len(data)/2)
    # ub = [2**16-1, len(data[0]), len(data)]
    # lb = [1, 0, 0]
    bounds = (lb, ub)
    popt, pcov = curve_fit(gauss_cross_corr, XY, data.ravel(), bounds=bounds, p0=init_guess, maxfev=1e5)
    gauss_val = gauss_cross_corr(XY, *popt)
    return *popt, gauss_val

def gauss_cross_corr_fit2(data):
    Y = np.arange(0,len(data))
    X = np.arange(0,len(data[0]))
    # central_pixel_x = np.round(np.median(X))
    # central_pixel_y = np.round(np.median(Y))
    #def gauss_spot_fixed_sigma(xy, A, x0, y0):
    def gauss_cross_corr(xy, A, x0, y0, sx, sy):
        (x, y) = xy
        return 2*A*np.e**(-(x-x0)**2/(2*sx**2) - (y-y0)**2/(2*sy**2))
    X, Y = np.meshgrid(X,Y)
    XY = np.vstack((X.ravel(), Y.ravel()))
    init_guess = (20, len(data[0])/2, len(data)/2, 2, 2)
    ub = [2**16-1, len(data[0]), len(data), 5, 5]
    lb = [0, 0, 0, 0, 0]
    # init_guess = (5000, len(data[0])/2, len(data)/2)
    # ub = [2**16-1, len(data[0]), len(data)]
    # lb = [1, 0, 0]
    bounds = (lb, ub)
    popt, pcov = curve_fit(gauss_cross_corr, XY, data.ravel(), bounds=bounds, p0=init_guess, maxfev=1e5)
    gauss_val = gauss_cross_corr(XY, *popt)
    return *popt, gauss_val

def FGCentroid2(M, pkRow, pkCol, Ncentr, Method, SNRthresh, sigmax_avg=None, sigmay_avg=None, *args):

    nr = len(M)
    nc = len(M[0])
    rowMin = max(0, pkRow - math.floor(Ncentr/2))
    rowMax = min(nr-1, pkRow + math.floor(Ncentr/2))
    colMin = max(0, pkCol - math.floor(Ncentr/2))
    colMax = min(nc-1, pkCol + math.floor(Ncentr/2))
    rowInds = list(np.arange(rowMin, rowMax+1))
    colInds = list(np.arange(colMin, colMax+1))
    # rowInds = [i for i in range(rowMin, rowMax + 1)]
    # colInds = [i for i in range(colMin, colMax + 1)]

    image = M[rowMin:rowMax+1, colMin:colMax+1]
    # image = []
    # for r in range(rowMin, rowMax+1):
    #     temp = []
    #     for c in range(colMin, colMax+1):
    #         temp.append(M[r][c])
    #     image.append(temp)

    x, y = np.meshgrid(np.arange(len(colInds)), np.arange(len(rowInds)))
    #N = [len(image[i]) for i in range(len(image))]

    centroid = {}
    if Method == 'FirstMoment':
        Msubtot = sumMat(image)
        xCentroid = sumMat(np.multiply(x, image))
        yCentroid = sumMat(np.multiply(y, image))
        centroid['row'] = yCentroid/Msubtot + rowMin
        centroid['col'] = xCentroid/Msubtot + colMin
        centroid['GauImg'] = 0 #SigEst2
        centroid['GauAmp'] = 0 #A2
        centroid['GauSig'] = 0 #np.sqrt(sigmax2**2 + sigmay2**2)
        centroid['GauSigx'] = 0 #sigmax2
        centroid['GauSigy'] = 0 #sigmay2
        centroid['offset'] = 0 #offset
    elif Method == 'Gaussian':
        if sigmax_avg is not None and sigmay_avg is not None:
            offset, A2, xc2, yc2, SigEst2 = gauss_fit_fixed_sigma(image, sigmax_avg, sigmay_avg)
            sigmax2 = sigmax_avg
            sigmay2 = sigmay_avg
        else:
            offset, A2, xc2, yc2, sigmax2, sigmay2, SigEst2 = gauss_fit(image)
        if np.isnan(float(xc2)) or np.isnan(float(yc2)) or float(xc2)<0 or float(yc2)<0 or float(xc2)>image.shape[1]-1 or float(yc2)>image.shape[0]-1 or \
        np.isnan(float(sigmax2)) or np.isnan(float(sigmay2)) or np.isnan(float(A2)):
            print("Even a true Gaussian fit gives NaNs")
        centroid['row'] = yc2 + rowMin
        centroid['col'] = xc2 + colMin
        centroid['GauImg'] = SigEst2
        centroid['GauAmp'] = A2
        centroid['GauSig'] = np.sqrt(sigmax2**2 + sigmay2**2)
        centroid['GauSigx'] = sigmax2
        centroid['GauSigy'] = sigmay2
        centroid['offset'] = offset

    elif Method == 'FastGaussian':
        # based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6163372/
        nSamp = 5 # The paper says should be at least 5; if this is too big, a light-skewed PSF could really throw off the lst_sq_fit() and give NaNs.
        # XXX make nSamp an input too? And change so that if SNR.max() < SNRthresh to go with half of SNR.max() as threshold?
        I, ui = maxk(flatten(image), nSamp)
        ux, uy = np.unravel_index(ui, (len(colInds), len(rowInds)))
        #XXX add a condition that ensures the 4 points closest to brightest one are included (in the case of ligh skew)?
        # brx = ux[-1]
        # bry = uy[-1]
        # uxx = np.append(ux, np.array([brx, brx+1, brx-1, brx, brx, brx-1, brx-1, brx+1, brx+1]))
        # uyy = np.append(uy, np.array([bry, bry, bry, bry+1, bry-1, bry-1, bry+1, bry-1, bry+1]))
        # II = image[uyy, uxx]
        # don't fit for intensities that are <= 0; since log of this will give nan
        bad_ind = np.where(I<=0)
        I = np.delete(I, bad_ind)
        ux = np.delete(ux, bad_ind)
        uy = np.delete(uy, bad_ind)

        xc1, yc1, sigmax1, sigmay1, A1 = lst_sq_fit(ux, uy, I)
        # rows, cols = np.where(image >= 0)
        # xc1, yc1, sigmax1, sigmay1, A1 = lst_sq_fit(cols, rows, image[rows,cols])
        if np.isnan(float(xc1)) or np.isnan(float(yc1)) or float(xc1)<0 or float(yc1)<0 or float(xc1)>image.shape[1]-1 or float(yc1)>image.shape[0]-1 or \
            np.isnan(float(sigmax1)) or np.isnan(float(sigmay1)) or np.isnan(float(A1)):
            pass
            # then do an actual nonlinear least squares fit for Guassian
            offset, A2, xc2, yc2, sigmax2, sigmay2, SigEst2 = gauss_fit(image)
            if np.isnan(float(xc2)) or np.isnan(float(yc2)) or float(xc2)<0 or float(yc2)<0 or float(xc2)>image.shape[1]-1 or float(yc2)>image.shape[0]-1 or \
            np.isnan(float(sigmax2)) or np.isnan(float(sigmay2)) or np.isnan(float(A2)):
                print("Even a true Gaussian fit gives NaNs")
        else:
            SigEst1 = calcSigEst2(A1, x, xc1, sigmax1, y, yc1, sigmay1)
            Noise = image - SigEst1
            SNR = np.abs(SigEst1/Noise)
            if SNR.max() < SNRthresh:
                SNRthresh = SNR.max()/2
                print("SNRthresh reduced")
            if SNRthresh < 1:
                print('SNRthresh 0')
            goodrows, goodcols = np.where(SNR >= SNRthresh)
            if len(goodrows) == 0:
                print('len(goodrows) = 0')
            xc2, yc2, sigmax2, sigmay2, A2  = lst_sq_fit(goodcols, goodrows, image[goodrows, goodcols])
            SigEst2 = calcSigEst2(A2, x, xc2, sigmax2, y, yc2, sigmay2)
            if np.isnan(float(xc2)) or np.isnan(float(yc2)) or float(xc2)<0 or float(yc2)<0 or float(xc2)>image.shape[1]-1 or float(yc2)>image.shape[0]-1:
                print('nan output parameters!') # testing purposes
                offset, A2, xc2, yc2, sigmax2, sigmay2, SigEst2 = gauss_fit(image)
                if np.isnan(float(xc2)) or np.isnan(float(yc2)) or float(xc2)<0 or float(yc2)<0 or float(xc2)>image.shape[1]-1 or float(yc2)>image.shape[0]-1 or \
                np.isnan(float(sigmax2)) or np.isnan(float(sigmay2)) or np.isnan(float(A2)):
                    print("Even a true Gaussian fit gives NaNs")

        centroid['row'] = yc2 + rowMin
        centroid['col'] = xc2 + colMin
        centroid['GauImg'] = SigEst2
        centroid['GauAmp'] = A2
        centroid['GauSig'] = np.sqrt(sigmax2**2 + sigmay2**2)
        centroid['GauSigx'] = sigmax2
        centroid['GauSigy'] = sigmay2
        centroid['offset'] = 0 # by assumption for FastGaussian

    else:
        raise Exception('unsupported centroiding method!')

    if len(args) == 0:
        centroid['y'] = centroid['row']
        centroid['x'] = centroid['col']
    elif len(args) == 2:
                #might be worth checking out this next part.  I haven't tested it yet, so there might be some issues...
        rowAxis = args[0]
        colAxis = args[1]
        rowLo = math.floor(centroid['row'])
        colLo = math.floor(centroid['col'])
        if rowLo < 0 or colLo < 0 or centroid['row'] >= rowMax or centroid['col'] >= colMax:
            raise Exception('This function is only designed to find centroids within the image provided. If the centroid is within the image, then the image may be undersampled.')
        centroid['y'] = rowAxis[rowLo] + (rowAxis[rowLo+1]-rowAxis[rowLo]) * (centroid['row'] - rowLo)
        centroid['x'] = colAxis[colLo] + (colAxis[colLo+1]-colAxis[colLo]) * (centroid['col'] - colLo)
    else:
        raise Exception('wrong number of arguments')


    return centroid


def FGCentroid3(pix_corr_x, pix_corr_y, M, pkRow, pkCol, Ncentr, Method, SNRthresh, bright_x, bright_y, sigmax_avg=None, sigmay_avg=None, *args):
    '''Takes into account corrected pixel positions for x and y (not z).'''
    nr = len(M)
    nc = len(M[0])
    rowMin = max(0, pkRow - math.floor(Ncentr/2))
    rowMax = min(nr-1, pkRow + math.floor(Ncentr/2))
    colMin = max(0, pkCol - math.floor(Ncentr/2))
    colMax = min(nc-1, pkCol + math.floor(Ncentr/2))
    rowInds = list(np.arange(rowMin, rowMax+1))
    colInds = list(np.arange(colMin, colMax+1))
    # rowInds = [i for i in range(rowMin, rowMax + 1)]
    # colInds = [i for i in range(colMin, colMax + 1)]

    image = M[rowMin:rowMax+1, colMin:colMax+1].ravel()
    x, y = np.meshgrid(np.arange(colMin, colMax+1), np.arange(rowMin, rowMax+1))
    pix_x_subframe = pix_corr_x[rowMin:rowMax+1, colMin:colMax+1]
    pix_y_subframe = pix_corr_y[rowMin:rowMax+1, colMin:colMax+1]
    x_corr = x.ravel() + pix_x_subframe.ravel()
    y_corr = y.ravel() + pix_y_subframe.ravel()
    pkRow_fix = pix_y_subframe[pkRow-rowMin, pkCol-colMin]
    pkCol_fix = pix_x_subframe[pkRow - rowMin, pkCol - colMin]
    pkRow = pkRow + pkRow_fix
    pkCol = pkCol + pkCol_fix


    if False: #XXX testing out masking
        mask = np.zeros_like(M[rowMin:rowMax+1, colMin:colMax+1])
        #mask[64:76, 88:99] = 1
        #mask[8:21, 14:20] = 1; mask[11:17, 10:23] = 1
        # y0 = 14; x0=17; radius=0#6
        # mask = np.zeros_like(M[rowMin:rowMax+1, colMin:colMax+1])
        # for i in np.arange(int(np.round(y0-radius)), int(np.round(y0+radius))+1):
        #     for j in np.arange(int(np.round(-np.sqrt(radius**2-(i-y0)**2)+x0)), int(np.round(np.sqrt(radius**2-(i-y0)**2)+x0))+1):
        #         mask[i,j]=1
        #mask[15:,20:] =1
        psf = M[rowMin:rowMax+1, colMin:colMax+1].copy()
        mask = np.zeros_like(psf)
        rows, cols  =np.where(psf < np.percentile(psf, 50))
        mask[rows, cols]=1
        im = np.ma.masked_array(psf, mask = mask)
        # mask[pkRow-rowMin-10:pkRow-rowMin+10, pkCol-colMin-10:pkCol-colMin+10] = 1
        # im = np.ma.masked_array(M[rowMin:rowMax+1, colMin:colMax+1], mask=mask)
        image = im[~im.mask].copy()
        pix_x_subframe = (pix_corr_x[rowMin:rowMax+1, colMin:colMax+1])[~im.mask]
        pix_y_subframe = (pix_corr_y[rowMin:rowMax+1, colMin:colMax+1])[~im.mask]
        x = x[~im.mask].copy()
        y = y[~im.mask].copy()
        x_corr = x.ravel() + pix_x_subframe.ravel()
        y_corr = y.ravel() + pix_y_subframe.ravel()


    # x_off = x_corr.min()
    # y_off = y_corr.min()
    # x_corr = x_corr - x_off
    # y_corr = y_corr - y_off


    #N = [len(image[i]) for i in range(len(image))]
    num_rows = rowMax+1 - rowMin
    num_cols = colMax+1 - colMin
    centroid = {}
    if Method == 'FirstMoment':
        Msubtot = sumMat(image)
        xCentroid = np.sum(x_corr*image) #sumMat(np.multiply(x_corr, image))
        yCentroid = np.sum(y_corr*image) #sumMat(np.multiply(y_corr, image))
        centroid['row'] = yCentroid/Msubtot #+ y_off #rowMin
        centroid['col'] = xCentroid/Msubtot #+ x_off #colMin
        # if np.sqrt((xCentroid/Msubtot-pkCol)**2+(yCentroid/Msubtot -pkRow)**2) > 4:
        #         centroid['col'] = pkCol
        #         centroid['row'] = pkRow
        centroid['GauImg'] = 0 #SigEst2
        centroid['GauAmp'] = 0 #A2
        centroid['GauSig'] = 0 #np.sqrt(sigmax2**2 + sigmay2**2)
        centroid['GauSigx'] = 0 #sigmax2
        centroid['GauSigy'] = 0 #sigmay2
        centroid['offset'] = 0 #offset

    elif Method == 'RotGaussian':
        if sigmax_avg is not None and sigmay_avg is not None:
            try:
                offset, A2, xc2, yc2, theta, SigEst2 = rot_gauss_fit_fixed_sigma_corr(image, x_corr, y_corr, sigmax_avg, sigmay_avg)
                sigmax2 = sigmax_avg
                sigmay2 = sigmay_avg
                if np.sqrt((xc2-pkCol)**2+(yc2-pkRow)**2) > 4:
                    xc2 = pkCol
                    yc2 = pkRow
            except:
                A2 = 0
                sigmax2 = 0
                sigmay2 = 0
                try:
                    xc2, yc2, prim_mm, sec_mm, A, B, offset, a, b, SigEst2 = Airy_1fit_corr(image, x_corr, y_corr)
                except:
                    xc2 = pkCol
                    yc2 = pkRow
                    SigEst2 = 0
                    offset = 0
            # if np.sqrt((xc2-pkCol)**2+(yc2-pkRow)**2) > 1:
            #     xc2 = pkCol
            #     yc2 = pkRow
        else:
            try:
                offset, A2, xc2, yc2, sigmax2, sigmay2, theta, SigEst2 = rot_gauss_fit_corr(image, x_corr, y_corr)
            except:
                A2 = 0
                sigmax2 = 0
                sigmay2 = 0
                try:
                    xc2, yc2, prim_mm, sec_mm, A, B, offset, a, b, SigEst2 = Airy_1fit_corr(image, x_corr, y_corr)
                except:
                    xc2 = pkCol
                    yc2 = pkRow
                    SigEst2 = 0
                    offset = 0
            #get_fov_odd_square(cent_x, cent_y, bright_x, bright_y, focal_length=1000, theta=0.25, pp=0.00376, num_from_center_input = 3)
            # if np.sqrt((xc2-pkCol)**2+(yc2-pkRow)**2) > 3:
            #     xc2, yc2, prim_mm, sec_mm, A, B, offset, a, b, SigEst2 = Airy_1fit_corr(image, x_corr, y_corr)
                # xc2 = pkCol
                # yc2 = pkRow
        # if np.isnan(float(xc2)) or np.isnan(float(yc2)) or float(xc2)<0 or float(yc2)<0 or float(xc2)>x_corr.max()-x_corr.(min) or float(yc2)>y_corr.max()-y_corr.min() or \
        # np.isnan(float(sigmax2)) or np.isnan(float(sigmay2)) or np.isnan(float(A2)):
        if np.isnan(float(xc2)) or np.isnan(float(yc2)) or float(xc2)<x_corr.min() or float(yc2)<y_corr.min() or float(xc2)>x_corr.max() or float(yc2)>y_corr.max() or \
        np.isnan(float(sigmax2)) or np.isnan(float(sigmay2)) or np.isnan(float(A2)):
            print("Even a true Gaussian fit gives NaNs")
        centroid['row'] = yc2 #+ y_off #+ rowMin
        centroid['col'] = xc2 #+ x_off #+ colMin
        centroid['GauImg'] = SigEst2
        centroid['GauAmp'] = A2
        centroid['GauSig'] = np.sqrt(sigmax2**2 + sigmay2**2)
        centroid['GauSigx'] = sigmax2
        centroid['GauSigy'] = sigmay2
        centroid['offset'] = offset

    elif Method == 'Gaussian':
        if sigmax_avg is not None and sigmay_avg is not None:
            try:
                offset, A2, xc2, yc2, SigEst2 = gauss_fit_fixed_sigma_corr(image, x_corr, y_corr, sigmax_avg, sigmay_avg)
                sigmax2 = sigmax_avg
                sigmay2 = sigmay_avg
                if np.sqrt((xc2-pkCol)**2+(yc2-pkRow)**2) > 4:
                    xc2 = pkCol
                    yc2 = pkRow
            except:
                A2 = 0
                sigmax2 = 0
                sigmay2 = 0
                try:
                    xc2, yc2, prim_mm, sec_mm, A, B, offset, a, b, SigEst2 = Airy_1fit_corr(image, x_corr, y_corr)
                except:
                    xc2 = pkCol
                    yc2 = pkRow
                    SigEst2 = 0
                    offset = 0
            # if np.sqrt((xc2-pkCol)**2+(yc2-pkRow)**2) > 1:
            #     xc2 = pkCol
            #     yc2 = pkRow
        else:
            try:
                offset, A2, xc2, yc2, sigmax2, sigmay2, SigEst2 = gauss_fit_corr(image, x_corr, y_corr)
            except:
                A2 = 0
                sigmax2 = 0
                sigmay2 = 0
                try:
                    xc2, yc2, prim_mm, sec_mm, A, B, offset, a, b, SigEst2 = Airy_1fit_corr(image, x_corr, y_corr)
                except:
                    xc2 = pkCol
                    yc2 = pkRow
                    SigEst2 = 0
                    offset = 0
            #get_fov_odd_square(cent_x, cent_y, bright_x, bright_y, focal_length=1000, theta=0.25, pp=0.00376, num_from_center_input = 3)
            # if np.sqrt((xc2-pkCol)**2+(yc2-pkRow)**2) > 3:
            #     xc2, yc2, prim_mm, sec_mm, A, B, offset, a, b, SigEst2 = Airy_1fit_corr(image, x_corr, y_corr)
                # xc2 = pkCol
                # yc2 = pkRow
        # if np.isnan(float(xc2)) or np.isnan(float(yc2)) or float(xc2)<0 or float(yc2)<0 or float(xc2)>x_corr.max()-x_corr.(min) or float(yc2)>y_corr.max()-y_corr.min() or \
        # np.isnan(float(sigmax2)) or np.isnan(float(sigmay2)) or np.isnan(float(A2)):
        if np.isnan(float(xc2)) or np.isnan(float(yc2)) or float(xc2)<x_corr.min() or float(yc2)<y_corr.min() or float(xc2)>x_corr.max() or float(yc2)>y_corr.max() or \
        np.isnan(float(sigmax2)) or np.isnan(float(sigmay2)) or np.isnan(float(A2)):
            print("Even a true Gaussian fit gives NaNs")
        centroid['row'] = yc2 #+ y_off #+ rowMin
        centroid['col'] = xc2 #+ x_off #+ colMin
        centroid['GauImg'] = SigEst2
        centroid['GauAmp'] = A2
        centroid['GauSig'] = np.sqrt(sigmax2**2 + sigmay2**2)
        centroid['GauSigx'] = sigmax2
        centroid['GauSigy'] = sigmay2
        centroid['offset'] = offset

    elif Method == 'GaussArcTan':
        if sigmax_avg is not None and sigmay_avg is not None:
            offset, A2, xc2, yc2, sigmax2, sigmay2, B, C, b, c, x1, y1, SigEst2 = gauss_arctan_fit_corr(image, x_corr, y_corr, sigmax_avg, sigmay_avg)
            sigmax2 = sigmax_avg
            sigmay2 = sigmay_avg
            # if np.sqrt((xc2-pkCol)**2+(yc2-pkRow)**2) > 1:
            #     xc2 = pkCol
            #     yc2 = pkRow
        else:
            offset, A2, xc2, yc2, sigmax2, sigmay2, B, C, b, c, x1, y1, SigEst2 = gauss_arctan_fit_corr(image, x_corr, y_corr)
            #get_fov_odd_square(cent_x, cent_y, bright_x, bright_y, focal_length=1000, theta=0.25, pp=0.00376, num_from_center_input = 3)

        # if np.isnan(float(xc2)) or np.isnan(float(yc2)) or float(xc2)<0 or float(yc2)<0 or float(xc2)>x_corr.max()-x_corr.(min) or float(yc2)>y_corr.max()-y_corr.min() or \
        # np.isnan(float(sigmax2)) or np.isnan(float(sigmay2)) or np.isnan(float(A2)):
        if np.isnan(float(xc2)) or np.isnan(float(yc2)) or float(xc2)<x_corr.min() or float(yc2)<y_corr.min() or float(xc2)>x_corr.max() or float(yc2)>y_corr.max() or \
        np.isnan(float(sigmax2)) or np.isnan(float(sigmay2)) or np.isnan(float(A2)):
            print("Even a true Gaussian fit gives NaNs")
        centroid['row'] = yc2 #+ y_off #+ rowMin
        centroid['col'] = xc2 #+ x_off #+ colMin
        centroid['GauImg'] = SigEst2
        centroid['GauAmp'] = A2
        centroid['GauSig'] = np.sqrt(sigmax2**2 + sigmay2**2)
        centroid['GauSigx'] = sigmax2
        centroid['GauSigy'] = sigmay2
        centroid['offset'] = offset
    # elif Method == 'FastGaussian':
    #     # based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6163372/
    #     nSamp = 5 # The paper says should be at least 5; if this is too big, a light-skewed PSF could really throw off the lst_sq_fit() and give NaNs.
    #     # XXX make nSamp an input too? And change so that if SNR.max() < SNRthresh to go with half of SNR.max() as threshold?
    #     I, ui = maxk(flatten(image), nSamp)
    #     ux, uy = np.unravel_index(ui, (len(colInds), len(rowInds)))
    #     #XXX add a condition that ensures the 4 points closest to brightest one are included (in the case of ligh skew)?
    #     # brx = ux[-1]
    #     # bry = uy[-1]
    #     # uxx = np.append(ux, np.array([brx, brx+1, brx-1, brx, brx, brx-1, brx-1, brx+1, brx+1]))
    #     # uyy = np.append(uy, np.array([bry, bry, bry, bry+1, bry-1, bry-1, bry+1, bry-1, bry+1]))
    #     # II = image[uyy, uxx]
    #     # don't fit for intensities that are <= 0; since log of this will give nan
    #     bad_ind = np.where(I<=0)
    #     I = np.delete(I, bad_ind)
    #     ux = np.delete(ux, bad_ind)
    #     uy = np.delete(uy, bad_ind)

    #     xc1, yc1, sigmax1, sigmay1, A1 = lst_sq_fit(ux, uy, I)
    #     # rows, cols = np.where(image >= 0)
    #     # xc1, yc1, sigmax1, sigmay1, A1 = lst_sq_fit(cols, rows, image[rows,cols])
    #     if np.isnan(float(xc1)) or np.isnan(float(yc1)) or float(xc1)<0 or float(yc1)<0 or float(xc1)>image.shape[1]-1 or float(yc1)>image.shape[0]-1 or \
    #         np.isnan(float(sigmax1)) or np.isnan(float(sigmay1)) or np.isnan(float(A1)):
    #         pass
    #         # then do an actual nonlinear least squares fit for Guassian
    #         offset, A2, xc2, yc2, sigmax2, sigmay2, SigEst2 = gauss_fit(image)
    #         if np.isnan(float(xc2)) or np.isnan(float(yc2)) or float(xc2)<0 or float(yc2)<0 or float(xc2)>image.shape[1]-1 or float(yc2)>image.shape[0]-1 or \
    #         np.isnan(float(sigmax2)) or np.isnan(float(sigmay2)) or np.isnan(float(A2)):
    #             print("Even a true Gaussian fit gives NaNs")
    #     else:
    #         SigEst1 = calcSigEst2(A1, x, xc1, sigmax1, y, yc1, sigmay1)
    #         Noise = image - SigEst1
    #         SNR = np.abs(SigEst1/Noise)
    #         if SNR.max() < SNRthresh:
    #             SNRthresh = SNR.max()/2
    #             print("SNRthresh reduced")
    #         if SNRthresh < 1:
    #             print('SNRthresh 0')
    #         goodrows, goodcols = np.where(SNR >= SNRthresh)
    #         if len(goodrows) == 0:
    #             print('len(goodrows) = 0')
    #         xc2, yc2, sigmax2, sigmay2, A2  = lst_sq_fit(goodcols, goodrows, image[goodrows, goodcols])
    #         SigEst2 = calcSigEst2(A2, x, xc2, sigmax2, y, yc2, sigmay2)
    #         if np.isnan(float(xc2)) or np.isnan(float(yc2)) or float(xc2)<0 or float(yc2)<0 or float(xc2)>image.shape[1]-1 or float(yc2)>image.shape[0]-1:
    #             print('nan output parameters!') # testing purposes
    #             offset, A2, xc2, yc2, sigmax2, sigmay2, SigEst2 = gauss_fit(image)
    #             if np.isnan(float(xc2)) or np.isnan(float(yc2)) or float(xc2)<0 or float(yc2)<0 or float(xc2)>image.shape[1]-1 or float(yc2)>image.shape[0]-1 or \
    #             np.isnan(float(sigmax2)) or np.isnan(float(sigmay2)) or np.isnan(float(A2)):
    #                 print("Even a true Gaussian fit gives NaNs")

    #     centroid['row'] = yc2 + rowMin
    #     centroid['col'] = xc2 + colMin
    #     centroid['GauImg'] = SigEst2
    #     centroid['GauAmp'] = A2
    #     centroid['GauSig'] = np.sqrt(sigmax2**2 + sigmay2**2)
    #     centroid['GauSigx'] = sigmax2
    #     centroid['GauSigy'] = sigmay2
    #     centroid['offset'] = 0 # by assumption for FastGaussian
    elif Method == 'astigmatic':
        #offset, A2, xc2, yc2, a, b, val = sincs_fit(image, x_corr, y_corr)
        #offset, A2, xc2, yc2, a, b, c, d, val = sincs_fit2(image, x_corr, y_corr)
        #offset, A2, xc2, yc2, a, b, c, d, e, f, val = sincs_fit3(image, x_corr, y_corr)
        #offset, A2, B2, xc2, yc2, a, b, c, d, val = sincs_fit4(image, x_corr, y_corr)
        out = sincs_5_fit_corr(image, x_corr, y_corr)
        x_coord = np.mean([out[4], out[8], out[12], out[16], out[20]])
        y_coord = np.mean([out[5], out[9], out[13], out[17], out[21]])
        # if np.isnan(float(xc2)) or np.isnan(float(yc2)) or float(xc2)<0 or float(yc2)<0 or float(xc2)>x_corr.max()-x_corr.(min) or float(yc2)>y_corr.max()-y_corr.min() or \
        # np.isnan(float(sigmax2)) or np.isnan(float(sigmay2)) or np.isnan(float(A2)):
        # if np.isnan(float(xc2)) or np.isnan(float(yc2)) or float(xc2)<x_corr.min() or float(yc2)<y_corr.min() or float(xc2)>x_corr.max() or float(yc2)>y_corr.max() or \
        # np.isnan(float(a)) or np.isnan(float(b)) or np.isnan(float(A2)):
        #     print("Even a true Gaussian fit gives NaNs")
        centroid['row'] = y_coord #yc2 #+ y_off #+ rowMin
        centroid['col'] = x_coord #xc2 #+ x_off #+ colMin
        centroid['GauImg'] = 0 #val #SigEst2
        centroid['GauAmp'] = 0 #A2
        centroid['GauSig'] = 0 # np.sqrt(sigmax2**2 + sigmay2**2)
        centroid['GauSigx'] = 0#a #sigmax2
        centroid['GauSigy'] = 0#b #sigmay2
        centroid['offset'] = out[0]

    elif Method == 'Gaussian5':
        out = gauss_5_fit_corr(image, x_corr, y_corr)
        x_coord = np.median([out[2], out[7], out[12], out[17], out[22]])
        y_coord = np.median([out[3], out[8], out[13], out[18], out[23]])
        centroid['row'] = y_coord #+ y_off #+ rowMin
        centroid['col'] = x_coord #+ x_off #+ colMin
        centroid['GauImg'] = out[-1] #SigEst2
        centroid['GauAmp'] = 0
        centroid['GauSig'] = 0 # np.sqrt(sigmax2**2 + sigmay2**2)
        centroid['GauSigx'] = 0 #sigmax2
        centroid['GauSigy'] = 0 #sigmay2
        centroid['offset'] = out[0]

    elif Method == 'GaussDiff':
        if sigmax_avg is not None and sigmay_avg is not None:
            offset, A, x0, y0, B, x1, y1, sx1, sy1, SigEst2 = gauss_fit_diff_fixed_sigma_corr(image, x_corr, y_corr, sigmax_avg, sigmay_avg)
            sigmax2 = sigmax_avg
            sigmay2 = sigmay_avg
        else:
            offset, A, x0, y0, sigmax2, sigmay2, SigEst2 = gauss_fit_corr(image, x_corr, y_corr)
        centroid['row'] = y0
        centroid['col'] = x0
        centroid['GauImg'] = SigEst2
        centroid['GauAmp'] = 0
        centroid['GauSig'] = np.sqrt(sigmax2**2 + sigmay2**2)
        centroid['GauSigx'] = sigmax2
        centroid['GauSigy'] = sigmay2
        centroid['offset'] = offset

    elif Method == 'Airy':
        x0, y0, x1, y1, prim_mm, sec_mm, A, B, offset, val = Airy_fit_corr(image, x_corr, y_corr)
        centroid['row'] = y0
        centroid['col'] = x0
        centroid['GauImg'] = val
        centroid['GauAmp'] = 0
        centroid['GauSig'] = 0#np.sqrt(sigmax2**2 + sigmay2**2)
        centroid['GauSigx'] = 0#sigmax2
        centroid['GauSigy'] = 0#sigmay2
        centroid['offset'] = 0#offset
    elif Method == 'Airy1':
        try:
            x0, y0, prim_mm, sec_mm, A, B, offset, a, b, val = Airy_1fit_corr(image, x_corr, y_corr)
        except:
            x0 = pkCol
            y0 = pkRow
            val = 0
        # if np.sqrt((x0-pkCol)**2+(y0-pkRow)**2) > 4: #XXX
        #         x0 = pkCol
        #         y0 = pkRow
        centroid['row'] = y0
        centroid['col'] = x0
        centroid['GauImg'] = val
        centroid['GauAmp'] = 0
        centroid['GauSig'] = 0#np.sqrt(sigmax2**2 + sigmay2**2)
        centroid['GauSigx'] = 0#sigmax2
        centroid['GauSigy'] = 0#sigmay2
        centroid['offset'] = 0#offset

    elif Method == 'Airy1Cont':
        x0, y0, prim_mm, sec_mm, A, B, offset, a, b, val = Airy_1fit_corr(image, x_corr, y_corr)
        if np.sqrt((x0-pkCol)**2+(y0-pkRow)**2) > 2:
            B, C, b, c, x0, y0, val = arctan_fit_corr(image, x_corr, y_corr)
        centroid['row'] = y0
        centroid['col'] = x0
        centroid['GauImg'] = val
        centroid['GauAmp'] = 0
        centroid['GauSig'] = 0#np.sqrt(sigmax2**2 + sigmay2**2)
        centroid['GauSigx'] = 0#sigmax2
        centroid['GauSigy'] = 0#sigmay2
        centroid['offset'] = 0#offset

    elif Method == 'Airy1arctan':
        x0, y0, prim_mm, sec_mm, A, B, offset, a, b, C, D, c, d, x1, y1, val = Airy_arctan_1fit_corr(image, x_corr, y_corr)
        # if np.sqrt((x0-pkCol)**2+(y0-pkRow)**2) > 4: #XXX
        #         x0 = pkCol
        #         y0 = pkRow
        centroid['row'] = y0
        centroid['col'] = x0
        centroid['GauImg'] = val
        centroid['GauAmp'] = 0
        centroid['GauSig'] = 0#np.sqrt(sigmax2**2 + sigmay2**2)
        centroid['GauSigx'] = 0#sigmax2
        centroid['GauSigy'] = 0#sigmay2
        centroid['offset'] = 0#offset


    elif Method == 'peak':
        centroid['row'] = pkRow
        centroid['col'] = pkCol
        centroid['GauImg'] = 0
        centroid['GauAmp'] = 0
        centroid['GauSig'] = 0#np.sqrt(sigmax2**2 + sigmay2**2)
        centroid['GauSigx'] = 0#sigmax2
        centroid['GauSigy'] = 0#sigmay2
        centroid['offset'] = 0#offset
    else:
        raise Exception('unsupported centroiding method!')

    if len(args) == 0:
        centroid['y'] = centroid['row']
        centroid['x'] = centroid['col']
    elif len(args) == 2:
                #might be worth checking out this next part.  I haven't tested it yet, so there might be some issues...
        rowAxis = args[0]
        colAxis = args[1]
        rowLo = math.floor(centroid['row'])
        colLo = math.floor(centroid['col'])
        if rowLo < 0 or colLo < 0 or centroid['row'] >= rowMax or centroid['col'] >= colMax:
            raise Exception('This function is only designed to find centroids within the image provided. If the centroid is within the image, then the image may be undersampled.')
        centroid['y'] = rowAxis[rowLo] + (rowAxis[rowLo+1]-rowAxis[rowLo]) * (centroid['row'] - rowLo)
        centroid['x'] = colAxis[colLo] + (colAxis[colLo+1]-colAxis[colLo]) * (centroid['col'] - colLo)
    else:
        raise Exception('wrong number of arguments')


    return centroid



if __name__ == "__main__":
    import os
    from astropy.io import fits
    import matplotlib.pyplot as plt
    import time

    #psf = np.load('psf_ex.npy',allow_pickle=True)
    #psf = fits.getdata(r'/Users/kevinludwick/D/CAO1/4_PixelMetrology/Lab data and FFT images/Hologram Data/11_15_22_500mm_2xtele_unaligned_pupil/fits_folder/PSF2020.fits')
    #psf = fits.getdata(r'/Users/kevinludwick/D/CAO1/4_PixelMetrology/Lab data and FFT images/Hologram Data/11_15_22_500mm_2xtele_unaligned_pupil/fits_folder/PSF1435.fits')
    #psf = fits.getdata(r'/Users/kevinludwick/D/CAO1/4_PixelMetrology/Lab data and FFT images/Hologram Data/11_15_22_500mm_2xtele_unaligned_pupil/fits_folder/PSF1264.fits')
    frame = fits.getdata(r'/Users/kevinludwick/D/CAO1/4_PixelMetrology/Lab data and FFT images/Hologram Data/06_02_24_500m_2xtele_ZWO/avg_lights_sub_nominal_.fits')
    psf = frame[1010:1300, 6250:6420]
    pk = np.unravel_index(np.argmax(psf), psf.shape)
    pix_corr_x = np.zeros((psf.shape[0],psf.shape[1]))
    #centroid = FGCentroid3(pix_corr_x, pix_corr_x, psf, pk[0], pk[1], psf.shape[0]*2, 'Airy1', SNRthresh = 8, bright_x=0, bright_y=0)
    centroid = FGCentroid3(pix_corr_x, pix_corr_x, psf, pk[0], pk[1], 40, 'RotGaussian', SNRthresh = 8, bright_x=0, bright_y=0)

    # actually, this is a saturated spot, so this won't work
    # frames_folder = 'D:\\CAOfiles\\CAO1\\4_PixelMetrology\\Lab data and FFT images\\hyperbolicity data\\spots_alignment\\9_23_22'
    # for file in os.listdir(frames_folder):
    #     f = os.path.join(frames_folder, file)
    #     if os.path.isfile(f) and f.endswith('.fits'):
    #         data = fits.getdata(f)
    #         hdul = fits.open(f)
    #         pk = np.unravel_index(np.argmax(data), data.shape)
    #         centroid = FGCentroid(data, pk[0], pk[1], 14, 'FirstMoment')
    #         break


    # def nCentroids_grid(M_in, n, m, Method, box_size):
    #     '''
    #     first will find the n highest peaks in the graph using nPeaks.
    #     Then for each peak, we find draw a box around that peak and calculate the centroid of the star in the box.
    #     This makes our list of centroids.
    #     '''
    #     y = nPeaks(M_in, n, m) #could take a long time
    #     print('nPeaks done')
    #     #cents = []
    #     cents_grid = np.zeros_like(M_in)
    #     for i in range(n):
    #         row = y[i][1]
    #         col = y[i][2]
    #         centroid = FGCentroid2(M_in, row, col, box_size, Method)

    #         cents_grid[(int(centroid['row']), int(centroid['col']))] = 1
    #         #cents.append({'row': centroid['row'], 'col': centroid['col']})
    #         pass

    #     return cents_grid
    path = 'D:\\CAOfiles\\CAO1\\4_PixelMetrology\\Lab data and FFT images\\Hologram Data\\'
    dir = '11_15_22_500mm_2xtele_unaligned_pupil' #unaligned: entrance pupil was about 15 mm above the center of curvature of the goniometer cradle
    darks = []
    lights = []
    for file in os.listdir(path+dir):
        f = os.path.join(path+dir, file)
        if 'nominal_dark_full_' in f:
            d = fits.getdata(f)
            darks.append(d.astype(float)) # to convert from unsigned integer uint16
        if 'nominal_full_' in f:
            df = fits.getdata(f)
            lights.append(df.astype(float))
    darks = np.stack(darks)
    lights = np.stack(lights)
    lights_subtracted = lights - darks
    avg_lights_sub = np.mean(lights_subtracted[2:], axis=0) #leaving off first 2, which are "transition" frames

    #test FastGaussian on saturated central PSF; as expected, doesn't work
    print(FGCentroid2(avg_lights_sub,3193, 4786, 55, 'FirstMoment', 8))

    ###########################
    f = 'D:\\CAOfiles\\CAO1\\4_PixelMetrology\\field_distortion\\holographic mask errors\\holographic\\Iimg_distorted.fits'
    d = fits.getdata(f)
    initial_t = time.time()
    #cents_grid = nCentroids_grid(d, 15*15, 8, 'FastGaussian', 14)
    cents_grid, cents, cent_x, cent_y = nCentroids_grid(d, 15*15, 8, 0, 2**16-1, 'FastGaussian', 8, 14, 5)
    print("time for nCentroids():  ", time.time() - initial_t)

    data = d[0:175, 0:175]
    pk = np.unravel_index(np.argmax(data), data.shape)
    centroid = FGCentroid2(data, pk[0], pk[1], 14, 'FastGaussian')

    # a case where I rotated an image using scipy.ndimage.rotate, and the splining involved smeared the PSFs
    f='D:\\CAOfiles\\CAO1\\4_PixelMetrology\\field_distortion\\holographic mask errors\\holographic\\rotated_centroid.fits'
    #XXX try FastGaussian on (110, 51) peak on 5th frame input: try including the 3x3 grid centered on the peak for the initial lst_sq_fit()
    # XXX when negative pixels around, FG initial fit bad; if that's true, maybe go to an exact non-log least-squares fit for Gaussian
    pass