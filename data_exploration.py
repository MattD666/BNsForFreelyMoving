# Exploring data and building preliminary structures for Bayesian networks
# Caleb Bessit
# 29 September 2025

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

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os

subjects = 10

# column 4

"""
For each feature, calculate the mean ± std on fr-trials vs nfr trials. Print out non-overlapping features.


"""
features = ['anterior P3 mean', 'anterior P3 STD', 'anterior alpha mean', 'alpha variability',
            'reaction time mean', 'reaction time variability', 'accuracy', 'log energy anterior P3',
            'shannon anterior P3', 'sure anterior P3', 'skewness anterior P3', 'kurtosis anteriro P3', 
            'log energy anterior alpha', 'shannon anterior alpha', 'sure anterior alpha', 'skewness anterior alpha', 
            'kurtosis anteriro alpha']

print(len(features))

# Relevant features
# Index 

plt.figure()

feat_fr_pos = np.arange(len(features))+1
feat_fr_pos, feat_nfr_pos = feat_fr_pos-0.1, feat_fr_pos+0.1


overall_fr, overall_nfr = None, None

for i in range(subjects):
    subject_id = f"{i+1:02d}"
    
    test = np.load(f"NP_Data/S1{subject_id}_data.npy")

    fr  = test[test[:,3] == 3.0][:,4:]
    nfr = test[test[:,3] == 4.0][:,4:]

    # Normalize arrays (because we'll compare the values later)
    mins, maxs = np.min(fr, axis=0), np.max(fr, axis=0)
    fr = (fr-mins)/(maxs-mins + 1e-20)

    mins, maxs = np.min(nfr, axis=0), np.max(nfr, axis=0)
    nfr = (nfr-mins)/(maxs-mins+ 1e-20)

    if overall_fr is None:
        overall_fr, overall_nfr = fr, nfr
    else:
        overall_fr  = np.vstack((overall_fr, fr))
        overall_nfr = np.vstack((overall_nfr, nfr))

    fr_mean, fr_std = np.mean(fr, axis=0), np.std(fr, axis=0)
    nfr_mean, nfr_std = np.mean(nfr, axis=0), np.std(nfr, axis=0)

    # for feature in features:
    #     print(f"    + {feature}: FR - {fr_mean} ± {fr_std} == NFR - {nfr_mean} ± {nfr_std}")

print(overall_fr.size, overall_nfr.size)
fr_mean, fr_std = np.mean(overall_fr, axis=0), np.std(overall_fr, axis=0)
nfr_mean, nfr_std = np.mean(overall_nfr, axis=0), np.std(overall_nfr, axis=0)

plt.errorbar(feat_fr_pos, np.abs(fr_mean), fr_std, ecolor='blue', capsize=4, fmt='o',linestyle='none',markersize=2)
plt.errorbar(feat_nfr_pos, np.abs(nfr_mean), nfr_std, ecolor='orange', capsize=4, fmt='o',linestyle='none',markersize=2)

# plt.plot(feat_fr_pos, np.abs(fr_mean), color='blue', linestyle='none', marker='o', label = "Freely-moving")
# plt.plot(feat_nfr_pos, np.abs(nfr_mean), color='orange', linestyle='none', marker='^', markerfacecolor='none', label='NOT freely-moving')

# Calculate the differences between free and non-free feature values, and print in order of features with largest normalized differences
diff = np.abs( np.abs(fr_mean) - np.abs(nfr_mean))

sorted_features = features
combined = list(zip(diff, sorted_features))
sorted_list = sorted(combined, key=lambda x:x[0], reverse=True)
diff, sorted_features = zip(*sorted_list)

for feature in range(len(sorted_features)):
    feature_name = sorted_features[feature]
    print(f" {feature+1}. {feature_name} ({diff[feature]}) [original index = {features.index(feature_name)+1}]")

plt.xlabel(f"Feature index")
plt.ylabel(f"Value ± std")
# plt.yscale('log')
plt.legend()
plt.title("Features and std")
plt.grid(alpha=0.3)
plt.show()
    # print(nfr.shape, fr.shape)
    # print(fr_mean.shape, fr_std.shape)
    # print(dict(Counter(test[:,3])))
    # print(f"+   Subject {subject_id}: {test.shape}")