import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

#number of channels
channel_list = [1,2,3]

#submit multiple channels 
for channel in channel_list:
    #write batch files
    job_file = f"dotdetection_channel{channel}.batch"
    with open(job_file, "w+") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --ntasks=1\n") #num tasks keep at 1
        f.write("#SBATCH --mem=50G\n") # RAM amount
        f.write("#SBATCH --cpus-per-task=12\n") #number of cpus
        f.write("#SBATCH --time=02:00:00\n") #how much time
        f.write("#SBATCH --array=0\n") #pos range
        f.write("pwd; hostname; date\n") #output path, hostname, date
        f.write("echo This is task $SLURM_ARRAY_TASK_ID\n") #output task id
        f.write("source ~/miniconda3/bin/activate\n") #activate source
        f.write("conda activate python3.7\n") #activate conda env
        f.write(f"python batch_dotdetection_individual.py {channel}\n") #run batch.py and feed in arguments
        f.write("date\n")
        f.close()

    os.system("sbatch %s" %job_file)
