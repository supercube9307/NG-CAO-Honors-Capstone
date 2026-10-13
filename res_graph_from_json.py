import json
import matplotlib.pyplot as plt
from im_filtering.helper_functions.file_io import *

json_file, path_input = get_directory_input("json",message = "Input path to strel_parameters.json: ", allow_folder=False)
json_path = os.path.join(path_input, json_file[0])

with open(json_path, "r") as file:
    json_dict = json.load(file)

res_array = [json_dict["pars_list"][x]["res"] for x in range(0,64)]

plt.bar(range(0,64), res_array)
plt.show() 