import os

def get_directory_input(expected_file_type="fits"):

    if expected_file_type[0] == ".":
        expected_file_type = expected_file_type.lstrip(".")

    file_names = []
    path_verified = False
    while path_verified == False:

        path_input = input(f"Input path to .{expected_file_type} file or folder containing .{expected_file_type} files: ")

        if not(os.path.exists(path_input)):
            print("Please select a valid file path")
            continue
        
        if os.path.isfile(path_input):
            print("Parsing File")
            print(path_input)
            path_input, file = path_input.rsplit("/",1)
            file_names.append(file)
            return(file_names, path_input)



        print("Parsing Folder")
        for file in os.listdir(path_input):
            file_names.append(file)

        for file in file_names:
            if file.split(".")[-1] != expected_file_type:
                print("Ingoring " + file)
                file_names.remove(file)

        if len(file_names) == 0:
            print(f"Please select a file or folder with .{expected_file_type} files")
            continue
        else:
            return(file_names, path_input)