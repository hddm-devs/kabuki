import kabuki
import pymc as pm
import numpy as np

# Set true values
true_mean = 2.
true_e1 = 1.
true_e2 = .5
true_e3 = -1.5
tau = 1/(2.**2) # Precision is inverse of variance.
tau_subj = 1.

# Generate subj means
subjs=15
true_subj_mean = pm.rnormal(true_mean, tau_subj, size=subjs)
true_subj_e1 = pm.rnormal(true_subj_mean+true_e1, 800)
true_subj_e2 = pm.rnormal(true_subj_mean+true_e2, 800)
true_subj_e3 = pm.rnormal(true_subj_mean+true_e3, 800)

# Generate data for each subj
N=50
samples=N*3
data = np.empty((subjs*samples), dtype=[('subj_idx','i4'), ('score','f4'), ('cond', 'S4')])
for subj in range(subjs):
    slice = data[subj*samples:(subj+1)*samples]
    slice['subj_idx'] = subj
    slice['score'][0:N] = pm.rnormal(true_subj_e1[subj], tau, size=N)
    slice['score'][N:2*N] = pm.rnormal(true_subj_e2[subj], tau, size=N)
    slice['score'][2*N:3*N] = pm.rnormal(true_subj_e3[subj], tau, size=N)
    slice[0:N]['cond'] = 'e1'
    slice[N:2*N]['cond'] = 'e2'
    slice[2*N:3*N]['cond'] = 'e3'
    
# Generate model
model = kabuki.models.Effect(data, is_subj_model=True, depends_on={'effect':['cond']})
model.mcmc(map_=False)
print model.summary()
