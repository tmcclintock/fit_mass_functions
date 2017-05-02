"""
This file contains the prior and likelihoods 
for doing the fits to create training data.
"""
import numpy as np

Tinker_defaults = {'d':1.97, 'e':1.0, "f": 0.51, 'g':1.228}

#Likelihood without scatter. Can run much faster
def lnlike_no_scatter(d,e,f,g,a,z,lM_bins,N_data,cov_data,icov_data,volume,MF_model):
    Len = len(a)
    LL = 0
    for i in range(Len):
        MF_model[i].set_parameters(d[i],e[i],f[i],g[i])
        N = MF_model[i].n_in_bins(lM_bins[i])*volume
        X = N_data[i] - N
        #LL+= -0.5*np.dot(X,np.dot(icov_data[i],X))
        cov = cov_data[i] + np.diag(0.0001*N**2)
        icov = np.linalg.inv(cov)
        LL+= -0.5*(np.dot(X,np.dot(icov,X)) +np.log(np.linalg.det(cov)))
    return LL

#Posterior
def lnprob(params,a,z,lM_bins,N_data,cov_data,icov_data,volume,MF_model):
    d0,d1,e0,e1,g0,g1 = params
    k = a-0.5
    d = d0 + k*d1
    e = e0 + k*e1
    f = np.ones_like(k)*Tinker_defaults['f']#f0 + k*f1
    g = g0 + k*g1
    if any(d<0) or any(d>5): return -np.inf
    if any(e<0) or any(e>5): return -np.inf
    if any(f<0) or any(f>5): return -np.inf
    if any(g<0) or any(g>5): return -np.inf
    return lnlike_no_scatter(d,e,f,g,a,z,lM_bins,N_data,cov_data,icov_data,volume,MF_model)
