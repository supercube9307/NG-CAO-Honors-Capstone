import numpy as np
import matplotlib.pyplot as plt

from im_filtering.helper_functions.file_io import *
from combined_filter import combined_filtering

KERNEL_SIZE = 15 # double check with original paper

def get_subframe(data: np.ndarray, y:int, x:int) -> np.ndarray:

    y_slice = slice(int(y - KERNEL_SIZE // 2), int(y - KERNEL_SIZE // 2 + KERNEL_SIZE))
    x_slice = slice(int(x - KERNEL_SIZE // 2), int(x - KERNEL_SIZE // 2 + KERNEL_SIZE))

    subframe = data[y_slice , x_slice]

    return subframe

def set_subframe(data: np.ndarray, y:int, x:int, subframe):

    y_slice = slice(int(y - KERNEL_SIZE // 2), int(y - KERNEL_SIZE // 2 + KERNEL_SIZE))
    x_slice = slice(int(x - KERNEL_SIZE // 2), int(x - KERNEL_SIZE // 2 + KERNEL_SIZE))

    data[y_slice , x_slice] = subframe

def get_median_subframe(frame, threshold = 2) -> np.ndarray:

    # ignore median pixels too close to edge
    frame_padded = frame[KERNEL_SIZE//2:-KERNEL_SIZE//2,KERNEL_SIZE//2:-KERNEL_SIZE//2]

    reference_value = np.median(frame_padded)
    median_coord_list = np.argwhere((frame_padded == reference_value)).tolist()
    if len(median_coord_list) == 0:
        median_coord_list = np.argwhere(np.isclose(frame_padded, reference_value)).tolist()

    frame_verified = False
    while frame_verified == False:

        median_coords = median_coord_list[len(median_coord_list) // 2]

        median_y = median_coords[0] + KERNEL_SIZE // 2
        median_x = median_coords[1] + KERNEL_SIZE // 2 

        median_subframe = get_subframe(frame, median_y, median_x)

        st_dev = np.std(median_subframe)
        mean = np.mean(median_subframe)

        outliers = (median_subframe - mean) / st_dev >  threshold
        if not outliers.any():
            frame_verified = True
        elif len(median_coord_list) == 1:
            frame_verified = True
        else:
            median_coord_list.remove(median_coords)

    return(median_subframe)

def find_PSFs(frame, psf_count):

    # make copy to erase psfs already chosen without destroying original data
    new_frame = frame.copy()
    # ignore PSFs too close to edge
    frame_padded = new_frame[KERNEL_SIZE//2:-KERNEL_SIZE//2,KERNEL_SIZE//2:-KERNEL_SIZE//2]

    psf_locs = []
    for idx in range(0, psf_count):
        psf_loc = np.argwhere(frame_padded == np.max(frame_padded))[0]
        psf_loc = [i + KERNEL_SIZE // 2 for i in psf_loc] # unpad the PSF Location
        psf_locs.append(psf_loc)

        psf_y = psf_loc[0]; psf_x = psf_loc[1]        

        y_slice = slice(int(psf_y - KERNEL_SIZE // 2), int(psf_y - KERNEL_SIZE // 2 + KERNEL_SIZE))
        x_slice = slice(int(psf_x - KERNEL_SIZE // 2), int(psf_x - KERNEL_SIZE // 2 + KERNEL_SIZE))

        new_frame[y_slice , x_slice] = np.zeros([KERNEL_SIZE, KERNEL_SIZE])

    # plt.figure()

    # plt.subplot(1,2,1)
    # plt.imshow(frame, vmax = 10000)
    # plt.title('Original Frame')

    # plt.subplot(1,2,2)
    # plt.imshow(new_frame, vmax = 10000)
    # plt.title('PSFs Identified')

    # plt.show()

    return(psf_locs)

def get_SNR(frame, median_subframe, psf_loc):
        
    psf_y = psf_loc[0]; psf_x = psf_loc[1]

    psf_subframe = get_subframe(frame, psf_y, psf_x)
    snr = np.sum(psf_subframe)/np.sum(median_subframe)

    return(snr, psf_subframe)

def show_PSFs_grid(subframes, title):

    psf_count = len(subframes)
    root = int(np.sqrt(psf_count))
    for i in range(root, 0, -1):
        if psf_count % i == 0:
            y_size, x_size = i, psf_count // i
            break

    for i in range(0, psf_count):
        plt.subplot(y_size, x_size, i + 1)
        plt.imshow(subframes[i], vmax = 10000)

    plt.suptitle(title)

    plt.show()

def main():

    psf_count = 20

    file_names, path_input = get_directory_input()
    file_path = os.path.join(path_input, file_names[0])

    raw_data  = read_fits_file(file_path)
    flt_data = combined_filtering(raw_data)

    raw_med_frame = get_median_subframe(raw_data)
    flt_med_frame = get_median_subframe(flt_data)

    # plt.figure()

    # plt.subplot(1,2,1)
    # plt.imshow(raw_med_frame, vmax = 10000)
    # plt.title('Raw Median Subframe')

    # plt.subplot(1,2,2)
    # plt.imshow(flt_med_frame, vmax = 10000)
    # plt.title('Filtered Median Subframe')

    psf_locs = find_PSFs(flt_data, psf_count)

    raw_SNRs = []
    flt_SNRs = []
    raw_PSF_subframes = []
    flt_PSF_subframes = []
    for psf_loc in psf_locs:

        [raw_SNR, raw_subframe] = get_SNR(raw_data, raw_med_frame, psf_loc)
        [flt_SNR, flt_subframe] = get_SNR(flt_data, flt_med_frame, psf_loc)

        raw_PSF_subframes.append(raw_subframe)
        flt_PSF_subframes.append(flt_subframe)
        raw_SNRs.append(raw_SNR)
        flt_SNRs.append(flt_SNR)

    show_PSFs_grid(raw_PSF_subframes, 'Raw PSFs')
    show_PSFs_grid(flt_PSF_subframes, 'Filtered PSFs')

    diff_SNRs = np.subtract(flt_SNRs, raw_SNRs)

    plt.figure()

    plt.subplot(3,1,1)
    plt.bar(range(0, psf_count), raw_SNRs)
    plt.title('Raw SNRs')

    plt.subplot(3,1,2)
    plt.bar(range(0, psf_count), flt_SNRs)
    plt.title('Filtered SNRs')

    plt.subplot(3,1,3)
    plt.bar(range(0, psf_count), diff_SNRs)
    plt.title('SNR Difference')

    plt.show()

if __name__ == "__main__":
    main()