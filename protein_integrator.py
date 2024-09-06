## LIBRARY CONFIGURATION

from cfg import *
sys.path.append('libraries')

from libraries.utils import *

import libraries.AllenRegistration_beta as AR_Lib
import libraries.AllenRegistration as AR_Lib_og


# FUNCTIONS
def fast_section_registration(section_id, AllenData, section_dict, experiment_dir):

    input_dict = {}
    input_dict[section_id] = AR_Lib.ImageRegistration(AllenData, section_dict[section_id], experiment_dir)
    input_dict[section_id].register_section()

    return input_dict


## SETTING THE PATHS TO BE USED BY THE PIPELINE

current_path = './' # to be replaced with os.getcwd()

# Choose directories for pre-process atlas files, flatmap files, input dataset and other code repositories
atlas_dir = 'atlas_files'
sys.path.append(atlas_dir)

flatmapper_dir = 'cortical_coordinates'
sys.path.append(flatmapper_dir)

dataset_dir = os.path.join(current_path, 'Datasets') # To be replaced with 'Datasets'
sys.path.append(dataset_dir)

# Choose a directory to save your output
img_output_dir = os.path.join(current_path,'Saves')
save_figure_dir = os.path.join(current_path,'Figures')

# Configuration parameters
resolution = 10 # Intiializing for 10 um resolution
save_registration = True
load_registration = False


#!!!! Parallel Approach !!!!!!!!!!!!!!!!!!!!

# Initializing the original constructor
AR_dict = OrderedDict()
experiment_list = os.listdir(dataset_dir)
AllenData = AR_Lib.AllenDataStore(flatmapper_dir, atlas_dir, resolution)

for experiment_id in experiment_list:

    experiment_dir = os.path.join(dataset_dir, experiment_id)
    section_dict = AR_Lib.get_section_info(experiment_dir)
    if section_dict == -1: continue

    partial_process_item = functools.partial(fast_section_registration, AllenData=AllenData, section_dict=section_dict,
                                             experiment_dir=experiment_dir)

    start_time = time.time()
    AR_dict[experiment_id] = Parallel(n_jobs=10)(delayed(partial_process_item)(section_id) for section_id in section_dict.keys())
    end_time = time.time()
    print(f'{experiment_id} was registered in {np.round((end_time - start_time) / 60).astype(int)} minutes')
