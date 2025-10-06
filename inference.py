# Test the utility of the classification network
# Caleb Bessit
# 05 October 2025

import os
import itertools
import numpy as np
import pandas as pd
import pyagrum as gum
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss



#Load network and test data
bn = gum.loadBN( os.path.join("bayesian_networks","FreelyMovingThoughts.bif") )

subject_range = (10,15)

# Specify indices of behavioural features 

original_features = ['p3_mean', 'p3_std', 'alpha_mean', 'alpha_var',
            'rt_mean', 'rt_var', 'ra', 'p3_log',
            'p3_shan', 'p3_sure', 'p3_skew', 'p3_kurt', 
            'alpha_log', 'alpha_shan', 'alpha_sure', 'alpha_skew', 
            'alpha_kurt']

alpha_features      = ['alpha_var','alpha_kurt','alpha_shan']
behavioral_features = ['rt_mean','rt_var','ra']
erp_features        = ['p3_kurt']

chosen_features = alpha_features + behavioral_features + erp_features

feat_index = dict( zip( chosen_features, [original_features.index(feature) for feature in chosen_features] ) )


overall_labels, overall_features = None, None
for subject in range(subject_range[0], subject_range[1]):
    subject_id = f"{subject+1:02d}"
    subject_data = np.load(f"NP_Data/S1{subject_id}_data.npy")

    print(subject_data.shape)

    if overall_labels is None:
        overall_labels   = subject_data[:,3]
        overall_features = subject_data[:,4:] 
    else:
        overall_labels   = np.concatenate( (overall_labels, subject_data[:,3] ))
        overall_features = np.vstack( (overall_features, subject_data[:,4:]))  

overall_labels = np.array([0 if label==3.0 else 1 for label in overall_labels])
trials = overall_labels.shape[0]
print(trials)

#################################
### INFERENCE
#################################

# Helper: run posterior given evidence dict (pyAgrum LazyPropagation)
def posterior_prob(target_var, evidence):
    ie = gum.LazyPropagation(bn)  # good for repeated single queries
    # set evidence expects dict varname->state (int)
    if evidence:
        ie.setEvidence(evidence)
    ie.makeInference()
    post = ie.posterior(target_var).toarray()  # returns array [P(0), P(1)]
    return float(post[1])  # return P(target=1)

def evidence_from_trial(i, subset):
    ev = {}
    row = overall_features[i]
    for feat in subset:
        ev[feat] = int(row[feat_index[feat]])
    return ev

def evaluate_subset_across_trials(subset):
    preds = []
    probs = []
    trues = overall_labels.tolist()
    eus = []
    best_actions = []
    for i in range(trials):
        ev = evidence_from_trial(i, subset)
        p1 = posterior_prob('freely_moving_thoughts', ev)
        prob = p1
        pred = 1 if prob >= 0.5 else 0
        preds.append(pred)
        probs.append(prob)
        # decision policy
        # act, eu = optimal_action_from_posterior(prob)
        # best_actions.append(act)
        # eus.append(eu)
    # metrics
    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, zero_division=0)
    rec = recall_score(trues, preds, zero_division=0)
    f1 = f1_score(trues, preds, zero_division=0)
    brier = brier_score_loss(trues, probs)
    # avg confidence: mean probability assigned to the chosen class
    chosen_confidences = []
    for p, pred in zip(probs, preds):
        chosen_confidences.append(p if pred==1 else (1.0-p))
    avg_confidence = float(np.mean(chosen_confidences))
    result = {
        'subset': subset,
        'n_trials': trials,
        'acc': acc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'brier': brier,
        'avg_confidence': avg_confidence,
    }
    #     'mean_prob_pos': float(np.mean(probs)),
    #     'prop_action1': float(np.mean(best_actions)),  # fraction of subjects where action=1 chosen
    # }
    return result, {'preds': preds, 'probs': probs} #, 'actions': best_actions, 'eus': eus}


results = []

# Do inference with no evidence

res_no_evidence, details_no = evaluate_subset_across_trials([])
res_no_evidence['label'] = 'no_evidence'
results.append(res_no_evidence)
print("No evidence:", res_no_evidence)

# Do inference using behavioural evidence only

beh_subsets = []
for i in range(1, len(behavioral_features)+1):
    for combination in itertools.combinations(behavioral_features, i):
        beh_subsets.append(tuple(combination))
print(f"Testing {len(beh_subsets)} behavioural subsets...")

beh_results = []
for subset in beh_subsets:
    result, _ = evaluate_subset_across_trials(list(subset))
    result['label'] = 'behavioural'
    results.append(result)
    beh_results.append(result)

# Do inference using physiological evidence only
phys_feats = alpha_features + erp_features
phys_subsets = []
for i in range(1, len(phys_feats)+1):
    for combinations in itertools.combinations(phys_feats, i):
        phys_subsets.append(tuple(combinations))
print(f"Testing {len(phys_subsets)} physiological subsets...")

phys_results = []
for subset in phys_subsets:
    result, _ = evaluate_subset_across_trials(list(subset))
    result['label'] = 'physiological'
    results.append(result)
    phys_results.append(result)

# Do inference using all evidence
res_all, details_all = evaluate_subset_across_trials(chosen_features)
res_all['label'] = 'all_features'
results.append(res_all)
print("All-features result:", res_all)

# Save results
outdir = "results"
os.makedirs(outdir, exist_ok=True)
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(['label','acc'], ascending=[True, False])
out_csv = "inference_results_summary.csv"
df_results.to_csv( os.path.join(outdir, out_csv), index=False)
print(f"Saved summary to {os.path.join(outdir, out_csv)}.")

# Utility: calculate optimal policies