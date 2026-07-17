import os

def output_csv_file(output_list, file_name="output.csv"):

    file_output_required = None
    while file_output_required != "y" and file_output_required != "n":

        file_output_required = input("Would you like to save to a file? (Y/N): ").lower()
        
        if file_output_required == "y" or file_output_required == "n":
            continue
        print("Please input a valid response")
            
    if file_output_required == "y":

        file_name = os.path.join("local_data",file_name)

        with open(file_name, "w") as output_file:

            output_text = ""
            for line in output_list:

                line = list(map(str, line))
                line_text = ",".join(line)+"\n"
                output_text += line_text

            output_file.write(output_text)

        print(f"File '{file_name}' created in " + os.getcwd())

def get_directory_input(expected_file_type="fits",message="") -> tuple:

    if expected_file_type[0] == ".":
        expected_file_type = expected_file_type.lstrip(".")

    file_names = []
    path_verified = False
    while path_verified == False:
        
        if message == "":
            message = f"Input path to .{expected_file_type} file or folder containing .{expected_file_type} files: "

        path_input = input(message)

        if not(os.path.exists(path_input)):
            error_path = os.path.join(os.getcwd(),path_input)
            print("Cannot Find: " + error_path)
            continue
        
        if os.path.isfile(path_input):
            print("Parsing File")
            path_input, file = path_input.rsplit("/",1)
            file_names.append(file)
            path_verified = True
            continue



        print("Parsing Folder")
        for file in os.listdir(path_input):
            file_names.append(file)


        removal_list = []
        for file in file_names:
            if file.split(".")[-1] != expected_file_type:
                removal_list.append(file)

        for file in removal_list:
            file_names.remove(file)
            print("Ingoring " + file)

        if len(file_names) == 0:
            print(f"Please select a file or folder with .{expected_file_type} files")
            continue

        path_verified = True
        
    return(file_names, path_input)