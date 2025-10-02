# Exploring data and building preliminary structures for Bayesian networks
# Original data available at https://osf.io/szvwa/files/osfstorage, use other script to convert to .npy files
# Caleb Bessit
# 29 September 2025

# Each file will have shape (S,21). Here, each file corresponds to a subject, the first dimension (S) is the number of samples for that participant,
# and the second dimension is the index for each piece of information.
# 
# According to the data documentation, below are each of the fields: 
# % Input features:
# % Column 1 to 3: participant information
# % Column 4:  experimental conditions
# % Column 5:  anterior P3 mean
# % Column 6:  anterior P3 STD
# % Column 7:  anterior alpha mean
# % Column 8: alpha variability
# % Column 9: reaction time mean
# % Column 10: reaction time variability
# % Column 11: accuracy
# % Column 12: log energy anterior P3
# % Column 13: shannon anterior P3
# % Column 14: sure anterior P3
# % Column 15: skewness anterior P3
# % Column 16: kurtosis anteriro P3
# % Column 17: log energy anterior alpha
# % Column 18: shannon anterior alpha
# % Column 19: sure anterior alpha
# % Column 20: skewness anterior alpha
# % Column 21: kurtosis anteriro alpha
# 
# Of interest to us is column 4, because that is a value which is equal to 3 or 4, where 3 = "fr" = "freely moving thoughts", and 4 = "nfr" = "not freely moving thoughts".

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os

# Can change the below up to 15
subjects = 10


features = ['anterior P3 mean', 'anterior P3 STD', 'anterior alpha mean', 'alpha variability',
            'reaction time mean', 'reaction time variability', 'accuracy', 'log energy anterior P3',
            'shannon anterior P3', 'sure anterior P3', 'skewness anterior P3', 'kurtosis anteriro P3', 
            'log energy anterior alpha', 'shannon anterior alpha', 'sure anterior alpha', 'skewness anterior alpha', 
            'kurtosis anteriro alpha']


plt.figure()

# Calculate the positions (on the graph) of where the information for each feature will go
feat_fr_pos = np.arange(len(features))+1
feat_fr_pos, feat_nfr_pos = feat_fr_pos-0.1, feat_fr_pos+0.1

# Aggregate data structure
overall_fr, overall_nfr = None, None


for i in range(subjects):
    subject_id = f"{i+1:02d}"
    
    # Load this subject's test file
    test = np.load(f"NP_Data/S1{subject_id}_data.npy")

    # Filter out the samples into those where they have freely moving thoughts (the value in column with index 3 is equal to 3.0) and 
    # those where they are NOT having freely moving thoughts (value in column with index 3 equal to 4.0)
    # Also only take the columns from index 4 onwards, because that corresponds to the feature columns. See the list of columns at the top of the file.
    fr  = test[test[:,3] == 3.0][:,4:]
    nfr = test[test[:,3] == 4.0][:,4:]

    # Normalize arrays (because we'll compare the values later)
    # By the way, I'm careful to normalize over the columns so that each feature is normalized with respect to it's own scale
    mins, maxs = np.min(fr, axis=0), np.max(fr, axis=0)
    fr = (fr-mins)/(maxs-mins + 1e-20)

    mins, maxs = np.min(nfr, axis=0), np.max(nfr, axis=0)
    nfr = (nfr-mins)/(maxs-mins+ 1e-20)

    # Concatenate to aggregate data structure
    if overall_fr is None:
        overall_fr, overall_nfr = fr, nfr
    else:
        overall_fr  = np.vstack((overall_fr, fr))
        overall_nfr = np.vstack((overall_nfr, nfr))

    
# Calculate the mean and std for each group of samples
fr_mean, fr_std = np.mean(overall_fr, axis=0), np.std(overall_fr, axis=0)
nfr_mean, nfr_std = np.mean(overall_nfr, axis=0), np.std(overall_nfr, axis=0)


# VERSION 1: Plot the values with error bars. Blue = fr, orange = nfr
plt.errorbar(feat_fr_pos, np.abs(fr_mean), fr_std, ecolor='blue', capsize=4, fmt='o',linestyle='none',markersize=2)
plt.errorbar(feat_nfr_pos, np.abs(nfr_mean), nfr_std, ecolor='orange', capsize=4, fmt='o',linestyle='none',markersize=2)

# VERSION 2: Just plot the values without error bars
# plt.plot(feat_fr_pos, np.abs(fr_mean), color='blue', linestyle='none', marker='o', label = "Freely-moving")
# plt.plot(feat_nfr_pos, np.abs(nfr_mean), color='orange', linestyle='none', marker='^', markerfacecolor='none', label='NOT freely-moving')

# Calculate the differences between free and non-free feature values, and print in order of features with largest normalized differences
diffs = np.abs( np.abs(fr_mean) - np.abs(nfr_mean))  #Worry about signs later

# Combine the features and their difference values, sort in descending order of magnitude of differences, and then unzip and print
combined = list(zip(diffs, features))
sorted_list = sorted(combined, key=lambda x:x[0], reverse=True)
sorted_diffs, sorted_features = zip(*sorted_list)

for feature in range(len(sorted_features)):
    feature_name = sorted_features[feature]
    print(f" {feature+1}. {feature_name} ({sorted_diffs[feature]}) [original index = {features.index(feature_name)+1}]")

plt.xlabel(f"Feature index")
plt.ylabel(f"Value ± std")
# plt.yscale('log')
plt.legend()
plt.title("Features and std")
plt.grid(alpha=0.3)
plt.show()