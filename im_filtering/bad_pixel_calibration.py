import im_filtering.helper_functions.fixedbp as fixedbp
from im_filtering.helper_functions.file_io import *

def pretty_output(output_list):
    
    if len(output_list) == 0:
        print("No Bad Pixels found within threshold")
        return
    
    for bp_loc in output_list:
        print(f"Bad Pixel found at: {bp_loc}")

def get_bool_2d_locs(bool_2d) -> list:

    x_len = len(bool_2d[0])
    y_len = len(bool_2d)
    output_list = []

    for y_index in range(y_len):
        for x_index in range(x_len):

            if bool_2d[y_index][x_index] == True:
                output_list.append([x_index,y_index])

    return(output_list)

def bp_correction(image: np.ndarray, bp_file_path = "local_data/Bad_Pixel_Calibration/bp_locations.csv", fwidth=16) -> np.ndarray:

    try:
        bp_locs = read_csv_file(bp_file_path)
    except:
        print("Calibration data not found in " + os.getcwd() + ". Regenerating " + bp_file_path)
        bp_detection(verbose=True)
        bp_locs = read_csv_file(bp_file_path)

    for bp_loc in bp_locs:
        bp_x = int(bp_loc[0])
        bp_y = int(bp_loc[1])

        subframe = image[bp_y-fwidth//2:bp_y+fwidth//2][bp_x-fwidth//2:bp_x+fwidth//2]

        sum = np.sum(subframe) - image[bp_y][bp_x]
        image[bp_y][bp_x] = sum/(subframe.size-1)

    return(image)

def bp_detection(verbose=False, bp_file_path = "local_data/Bad_Pixel_Calibration/bp_locations.csv"):

    frame_types = ["dark", "flat"]
    file_paths = {}
    for frame_type in frame_types:
        message = "\nPlease input path to .fits file containing master " + frame_type + " frame: "

        file_names, path_input = get_directory_input(message = message, allow_folder=False)
        file_paths[frame_type] =  os.path.join(path_input,file_names[0])

    dark_frame = read_fits_file(file_paths["dark"])
    flat_frame = read_fits_file(file_paths["flat"])

    bp_map = fixedbp.compute_fixedbp_excam(dark=dark_frame, flat=flat_frame, dthresh=5, ffrac=0.5, fwidth=16)

    bp_locs =  get_bool_2d_locs(bp_map)

    if verbose:
        pretty_output(bp_locs)

    write_csv_file(bp_locs, file_name=bp_file_path)

def main():

    print('\nGenerating calibration file')

    bp_detection(verbose=True)

if __name__ == '__main__':
    main()