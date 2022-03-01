from deepcell_spots_detection import find_spots_all
from pathlib import Path
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

directory = Path("/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/notebook_pyfiles/dapi_aligned/fiducial_aligned/")
position_name = 'MMStack_Pos{pos}.ome.tif'

files, _, _ = find_matching_files(directory, f'HybCycle_{JOB_ID}' + f'/{position_name}')
files = [str(f) for f in files]
files = files[0:2]

#This list should contain all hybcycles for a given pos
img_list = files
#Probability threshold to be considered a spot (parameter of deep cell)
probability_threshold = 0.85
#Cutoff for dot size beyond size_cutoff*std (from fitted gaussian)
size_cutoff = 4
#Path for output folder. Code will automatically make the folder.
output_folder = "/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/notebook_pyfiles/dots_detected/deepcell"
#If your data set is within channel encoded then set this true.
encoded_within_channel = True
#if performing job by hyb set this to true for file organization
job_by_hyb = True


find_spots_all(img_list, probability_threshold, size_cutoff, output_folder, encoded_within_channel, job_by_hyb)