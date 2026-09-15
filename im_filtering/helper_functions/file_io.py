import os
import numpy as np
from astropy.io import fits

def ask_file_output() -> bool:

    response = None
    while response != "y" and response != "n":

        response = input("Would you like to save to a file? (Y/N): ").lower()
        
        if response == "y" or response == "n":
            continue
        print("Please input a valid response")

    if response == "y":
        file_output_required = True
    else:
        file_output_required = False

    return(file_output_required)

def prep_file_write(file_name, file_output_required):

    # if file output is not mandatory, as user if they would like to save to a file
    if file_output_required == False:

        file_output_required = ask_file_output()

    # add 'local_data/' to file_name if not present
    if file_name.split(os.path.sep)[0] != "local_data":
        file_path = os.path.join("local_data",file_name)
    else:
        file_path = file_name

    return(file_path, file_output_required)

def write_fits_file(array: np.ndarray, file_name: str, file_output_required=False):

    file_path, file_output_required = prep_file_write(file_name, file_output_required)

    if file_output_required:

        hdu = fits.ImageHDU(array)
        prim = fits.PrimaryHDU()
        hdu_list = fits.HDUList([prim,hdu])
        
        hdu_list.writeto(file_path, overwrite=True)

        print(f"Wrote to: " + os.path.join(os.getcwd(), file_path))

def write_csv_file(output_list, file_name="output.csv",file_output_required=False):

    file_path, file_output_required = prep_file_write(file_name, file_output_required)
            
    if file_output_required:

        with open(file_path, "w") as output_file:

            output_text = ""
            for line in output_list:

                line = list(map(str, line))
                line_text = ",".join(line)+"\n"
                output_text += line_text

            output_file.write(output_text)

        print(f"Wrote to: " + os.path.join(os.getcwd(), file_path))

def read_fits_file(file_path: str) -> np.ndarray:

    with fits.open(file_path) as hdul:
        image = hdul[1].data #type: ignore

    return(image)

def read_csv_file(file_path: str) -> np.ndarray:

    with open(file_path) as file:
        text = file.read()

    line_list = text.splitlines()
    out_list = []

    for entry in line_list:
        out_list.append(entry.split(","))

    out_list = np.array(out_list)

    return(out_list)

def check_file(path_input :str) -> tuple:

    if os.path.isfile(path_input):

        path_input, file = path_input.rsplit("/",1)
        
        path_verified = True
        return(path_input, file, path_verified)
    
    else:
        return()

def check_folder(path_input: str, expected_file_type: str) -> tuple:

    if os.path.isdir:
        file_names = []
        for file in os.listdir(path_input):
            file_names.append(file)

        removal_list = []
        for file in file_names:
            if file.split(".")[-1] != expected_file_type:
                removal_list.append(file)

        for file in removal_list:
            file_names.remove(file)
            print("Ingoring " + file)

        path_verified = True
        return(path_input, file, path_verified)
    
    else:
        return()

def get_directory_input(expected_file_type="fits",message="", allow_folder=False) -> tuple:

    if expected_file_type[0] == ".":
        expected_file_type = expected_file_type.lstrip(".")

    file_names = []
    path_verified = False
    while path_verified == False:
        
        if message == "":
            
            message = f"Input path to .{expected_file_type} file"

            if allow_folder == True:
                message += f" or folder containing .{expected_file_type} files: "
            else:
                message += ": "

        path_input = input(message)

        if not(os.path.exists(path_input)):
            error_path = os.path.join(os.getcwd(),path_input)
            print("Cannot Find: " + error_path)
            continue

        if allow_folder == False:
            print("Parsing File")
            try:
                path_input, file, path_verified = check_file(path_input)
                file_names.append(file)
            except: pass

        if allow_folder == True:
            print("Parsing Folder")
            try: 
                path_input, file_names, path_verified = check_folder(path_input, expected_file_type)
            except: pass

        if len(file_names) != 0:
            path_verified = True
        else:
            print(f"Please make a valid selection")
            continue
        
    return(file_names, path_input)