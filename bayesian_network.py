# Bayesian network for detecting freely-moving thoughts.
# Caleb Bessit
# 02 October 2025

from pylab import *
import pyagrum as gum

# Create network
bn = gum.BayesNet('FreelyMovingThoughts')

# Add variables
'''
The 6 features with the largest normalized differences between freely-moving/non-freely-moving samples are:

    1. P3 mean
    2. sure alpha
    3. P3 STD
    4. sure P3
    5. reaction time variability
    6. shannon P3


'''

# Names of variables, structures as features + target
names = ['p3_mean', 'alpha_sure','p3_std', 'p3_sure','rt_var','p3_shannon'] + ['freely moving?']

id_p3_mean, id_alpha_sure, id_p3_std, id_p3_sure, id_rt_var, id_p3_shannon, id_freely_moving = [bn.add(name, 2) for name in names]

# Create arcs
for link in [
    (id_p3_mean, id_freely_moving),
    (id_alpha_sure, id_freely_moving),
    (id_p3_std, id_freely_moving),
    (id_p3_sure, id_freely_moving),
    (id_rt_var, id_freely_moving),
    (id_p3_shannon, id_freely_moving),
]:
    bn.addArc(*link)

print(bn)

# Topology created, now create probability tables.

# Individual variables
bn.cpt("p3_mean").fillWith([0.5,0.5])
bn.cpt("alpha_sure").fillWith([0.5,0.5])
bn.cpt("p3_std").fillWith([0.5,0.5])
bn.cpt("p3_sure").fillWith([0.5,0.5])
bn.cpt("rt_var").fillWith([0.5,0.5])
bn.cpt("p3_shannon").fillWith([0.5,0.5])

# Would have to cater for 2{number of variables}=2^6=64 cases if we treat it as all features linked to fr-node directly
# Will maybe assume some relationship between P3 variables