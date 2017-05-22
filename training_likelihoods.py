"""
This file contains the prior and likelihoods 
for doing the fits to create training data.
"""
import numpy as np
import sys

Tinker_defaults = {'d':1.97, 'e':1.0, "f": 0.51, 'g':1.228}

#Likelihood without scatter. Can run much faster
def lnlike_no_scatter(d,e,f,g,B,a,z,lM_bins,N_data,cov_data,icov_data,volume,MF_model):
    Len = len(a)
    LL = 0
    for i in range(Len):
        MF_model[i].set_parameters(d[i],e[i],f[i],g[i],B[i])
        N = MF_model[i].n_in_bins(lM_bins[i])*volume
        X = N_data[i] - N
        icov = icov_data[i]
        #cov = cov_data[i] + np.diag(0.0001*N**2)
        #icov = np.linalg.inv(cov)
        LL+= -0.5*np.dot(X,np.dot(icov,X))
    return LL

#Posterior
def lnprob(params, a, z, lM_bins, N_data, cov_data, icov_data, 
           volume, MF_model, name, Tinker_defaults):
    k = a-0.5
    B = [None for ai in a]
    if name == 'dfgB':
        d0,d1,f0,f1,g0,g1,B = params
        B = np.array([B for ai in a])
        e0 = Tinker_defaults['e']
        e1 = 0.0
    if name == 'defg':
        d0,d1,e0,e1,f0,f1,g0,g1 = params
    elif name == 'def':
        d0,d1,e0,e1,f0,f1 = params
        g0 = Tinker_defaults['g']
        g1 = 0.0
    elif name == 'deg':
        d0,d1,e0,e1,g0,g1 = params
        f0 = Tinker_defaults['f']
        f1 = 0.0
    elif name == 'dfg':
        d0,d1,f0,f1,g0,g1 = params
        e0 = Tinker_defaults['e']
        e1 = 0.0
    #d0,d1,e0,e1,f0,f1,g0,g1 = params
    d = d0 + k*d1
    e = e0 + k*e1
    f = f0 + k*f1
    g = g0 + k*g1
    if any(B):
        if any(B<0) or any(B>100): return -np.inf
    if any(d<0) or any(d>5): return -np.inf
    if any(e<0) or any(e>5): return -np.inf
    if any(f<0) or any(f>5): return -np.inf
    if any(g<0) or any(g>5): return -np.inf
    return lnlike_no_scatter(d,e,f,g,B,a,z,lM_bins,N_data,cov_data,icov_data,volume,MF_model)
