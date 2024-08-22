from daostarfinder_dotdetection import dot_detection
from pathlib import Path
import os
import sys
import numpy as np
from util import find_matching_files
import pandas as pd

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)
print(f'This is task {JOB_ID}')

#get channel info
channel = int(sys.argv[1])

#path to processed images
directory = Path("/path/to/data/pyfish_tools/output/pre_processed_images")
#jobs are submitted per FOV
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

#each row is channel and each column is hybs.
threshold_list = np.array([[250, 400, 400, 250, 250, 250, 250, 300, 300, 400, 250, 350, 400, 300, 300, 300, 300, 350, 300, 250],
                           [200, 200, 200, 250, 250, 250, 200, 300, 200, 200, 200, 250, 250, 250, 250, 500, 200, 250, 300, 400]])

#Number of channels in your image
num_channels=3 
# sigma cutoff for size distribution
size_cutoff = 4 

#looping through each hyb for a given channel on a given position
dots_list = []
for hyb in range(threshold_list.shape[1]):
    file = directory / f"HybCycle_{hyb}" / position_name
    threshold = threshold_list[channel-1, hyb] # threshold to detect all dots (if image was scaled)
    dots = dot_detection(file, HybCycle=hyb, size_cutoff=size_cutoff, threshold=threshold, channel=channel, num_channels=num_channels)
    if dots.empty:
        # If the DataFrame is empty, create a row with zeros and necessary info
        dots = pd.DataFrame({
            "hyb": [hyb],
            "ch": [channel],
            "x": [0],
            "y": [0],
            "z": [0],
            "flux": [0],
            "max intensity": [0],
            "sharpness": [0],
            "symmetry": [0],
            "roundness by gaussian fits": [0],
            "size": [0]})
    dots_list.append(dots)

#set output paths
parent = directory.parent
output_folder = parent / f"dots_detected/Channel_{channel}" 
output_folder.mkdir(parents=True, exist_ok=True)

#concat df
combined_df = pd.concat(dots_list).reset_index(drop=True)

#get number of z's
num_z = combined_df["z"].unique()

#get pos info
pos = position_name.split("_")[1].replace(".ome.tif","")

#output files
for z in num_z:
    combined_df_z = combined_df[combined_df["z"]==z]
    output_path = output_folder / pos
    output_path.mkdir(parents=True, exist_ok=True)
    combined_df_z.to_csv(str(output_path) +f"/locations_z_{int(z)}.csv", index=False)

