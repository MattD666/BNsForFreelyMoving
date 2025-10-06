# Converts raw data to evidence logits which can be used by the network for inference
# 06 October 2025
# Caleb Bessit

import os
import numpy as np

subjects = (10,15)

outdir = "test_bin_digits"
os.makedirs(outdir, exist_ok=True)
for subject in range(subjects[0], subjects[1]):
    subject_id = f"{subject+1:02d}"
    data = np.load(f"NP_Data/S1{subject_id}_data.npy")

    fr  = data[data[:,3] == 3.0][:,4:]
    nfr = data[data[:,3] == 4.0][:,4:]

    features = data[:,4:]

    fr_medians, nfr_medians = np.median(fr, axis=0), np.median(nfr, axis=0)
    fr_diffs, nfr_diffs = np.abs(features-fr_medians), np.abs(features-nfr_medians)
    
    bin_digits = (fr_diffs < nfr_diffs).astype(int)
    combined_data = np.hstack( (data[:,:4], bin_digits))

    np.save( os.path.join(outdir, f"S1{subject_id}_binary_digit_data.npy"), combined_data )
    print(f"Done with {subject_id}.")
    # print(combined_data.shape)


    # Code below is to manually check correctness
    # if subject==10:
    #     print(f"Data:\n\t= {data[:1,:]}\n")
    #     print(f"FR, NFR medians: \n\t+ {fr_medians}, \n\t- {nfr_medians}\n")
    #     print(f"Features: \n\t> {features[:1,:]}\n")
    #     print(f"Diffs: \n\t+ {fr_diffs[:1,:]}, \n\t- {nfr_diffs[:1,:]}\n")
    #     print(f"Logits: \n\t~ {logits[:1,:]}\n")
    #     print(f"Combined: \n\t= {combined_data[:1,:]}\n")



    
