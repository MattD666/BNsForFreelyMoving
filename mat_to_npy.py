# Converts .mat subject data files (from dataset) to Python .npy files
# Data can be found at https://osf.io/szvwa/files/osfstorage
# Caleb Bessit
# 29 September 2025

import scipy.io as sio
import numpy as np
import os

PATH = "Data/"

subjects = 15

out_path = "NP_Data/"
os.makedirs(out_path, exist_ok=True)

for i in range(subjects):
    subject_id = f"{i+1:02d}"
    print(subject_id)
    
    mat_data = sio.loadmat(f"Data/S1{subject_id}_data.mat")
    subject_data = mat_data['data_ml']
    print(np.array(subject_data).shape)
    np.save(f"NP_data/S1{subject_id}_data.npy", np.array(subject_data))
    print(f"    + Saved {subject_id}!\n")