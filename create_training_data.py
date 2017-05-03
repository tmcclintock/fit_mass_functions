import numpy as np
import scipy.optimize as op
import emcee, sys, corner, os
import visualize
import tinker_mass_function as TMF
import training_likelihoods as TL
import matplotlib.pyplot as plt

################################################
#CREATE THE OUTPUT FILES FROM SCRATCH OR NO?
from_scratch = False
################################################

#Choose which modes to run
run_test = False
run_best_fit = True
run_mcmc = True
run_mcmc_comparisons = False
calculate_chi2 = False

#MCMC configuration
nwalkers, nsteps = 16, 10000
nburn = 1000

#Scale factors, redshifts, volume
scale_factors = np.array([0.25, 0.333333, 0.5, 0.540541, 0.588235, 
                          0.645161, 0.714286, 0.8, 0.909091, 1.0])
redshifts = 1./scale_factors - 1.0
volume = 1050.**3 #[Mpc/h]^3

#Gather the cosmological parameters
N_boxes = 39 #39th is broken
N_z = 10 
cosmologies = np.genfromtxt("cosmos.txt")

#The paths to the data and covariances. This is hard coded in.
data_path = "N_data/Box%03d/Box%03d_Z%d.txt"
cov_path  = "N_data/Box%03d/Box%03d_cov_Z%d.txt"

#This contains our parameterization
name = 'defg'
corner_labels = []
header = ""
for i,l in zip(range(len(name)), name):
    corner_labels.append(r"$%s_0$"%l)
    corner_labels.append(r"$%s_1$"%l)
    header += "%s0\t"%l
    header += "%s1\t"%l
header +="\n"
N_parameters = len(corner_labels)
N_parameters = len(corner_labels)
Tinker_defaults = {'d':1.97, 'e':1.0, "f": 0.51, 'g':1.228}
#guesses = np.array([2.13, 0.11, 1.1, 0.2, 1.25, 0.11]) #d0,d1,e0,e1,g0,g1
guesses = np.array([2.13, 0.11, 1.1, 0.2, 0.41, 0.15, 1.25, 0.11]) #d0,d1,e0,e1,f0,f1,g0,g1
Tinker_defaults = {'d':1.97, 'e':1.0, "f": 0.51, 'g':1.228}
def get_params(model, sf):
    if name is 'defg':
        d0,d1,e0,e1,f0,f1,g0,g1 = model
    if name is 'def':
        d0,d1,e0,e1,f0,f1 = model
        g0 = Tinker_defaults['g']
        g1 = 0.0
    if name is 'deg':
        d0,d1,e0,e1,g0,g1 = model
        f0 = Tinker_defaults['f']
        f1 = 0.0
    if name is 'dfg':
        d0,d1,f0,f1,g0,g1 = model
        e0 = Tinker_defaults['e']
        e1 = 0.0
    k = sf - 0.5
    d = d0 + k*d1
    e = e0 + k*e1
    f = f0 + k*f1
    g = g0 + k*g1
    return d,e,f,g

#Create the output files
base_dir = "output/%s/"%name
os.system("mkdir -p %s"%base_dir)
base_save = base_dir+"%s_"%name
if from_scratch:
    best_fit_models = np.zeros((N_boxes,N_parameters))
    np.savetxt(base_save+"bests.txt",best_fit_models)
    mean_models = np.zeros((N_boxes,N_parameters))
    np.savetxt(base_save+"means.txt",mean_models)
    var_models = np.zeros((N_boxes,N_parameters))
    np.savetxt(base_save+"vars.txt",var_models)
    chi2s = np.zeros((N_boxes,N_z))
    np.savetxt(base_save+"BFchi2s.txt",chi2s)
else: 
    best_fit_models = np.loadtxt(base_save+"bests.txt")
    mean_models = np.loadtxt(base_save+"means.txt")
    var_models = np.loadtxt(base_save+"vars.txt")
    chi2s = np.loadtxt(base_save+"BFchi2s.txt")

#Loop over cosmologies and redshifts
box_lo,box_hi = 0,39
z_lo,z_hi = 0,10
for i in xrange(box_lo,box_hi):
    #Get in the cosmology and create a cosmo_dict
    num,ombh2,omch2,w0,ns,ln10As,H0,Neff,sigma8 = cosmologies[i]
    h = H0/100.
    Ob,Om = ombh2/(h**2), ombh2/(h**2)+omch2/(h**2)
    cosmo_dict = {"om":Om, "ob":Ob, "ol":1-Om, "ok":0.0, "h":h, 
                  "s8":sigma8, "ns":ns, "w0":w0, "wa":0.0}

    #Read in all of the mass functions
    lM_array = []
    lM_bin_array = []
    N_data_array = []
    cov_array = []
    icov_array = []
    TMF_array = []
    for j in xrange(0,N_z):
        #Get in the data
        lM_low,lM_high,N_data,NP = np.loadtxt(data_path%(i,i,j)).T
        cov = np.loadtxt(cov_path%(i,i,j))
        lM_bins = np.array([lM_low,lM_high]).T
        icov = np.linalg.inv(cov)
        #Add things to the arrays
        lM_array.append(np.log10(np.mean(10**lM_bins,1)))
        lM_bin_array.append(lM_bins)
        N_data_array.append(N_data)
        cov_array.append(cov)
        icov_array.append(icov)
        TMF_model = TMF.tinker_mass_function(cosmo_dict,redshifts[j])
        TMF_array.append(TMF_model)
        continue
    
    if run_test:
        test = TL.lnprob(guesses,scale_factors,redshifts,lM_bin_array,
                         N_data_array,cov_array,icov_array,volume,TMF_array,
                         name, Tinker_defaults)
        print "Test result = %f\n"%test

    if run_best_fit:
        nll = lambda *args:-TL.lnprob(*args)
        result = op.minimize(nll,guesses,
                             args=(scale_factors,redshifts,lM_bin_array,
                                   N_data_array,cov_array,icov_array,volume,
                                   TMF_array, name, Tinker_defaults),
                             method="Powell")
        best_fit_models[i] = result['x']
        print "Best fit for Box%03d:\n%s\n"%(i,result)
          
    if run_mcmc:
        ndim = N_parameters
        start = best_fit_models[i]
        print start
        pos = [start + 1e-3*np.random.randn(ndim) for k in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, TL.lnprob,
                                        args=(scale_factors,redshifts,lM_bin_array,
                                              N_data_array,cov_array,icov_array,volume,
                                              TMF_array, name, Tinker_defaults),threads = 8)
        print "Performing MCMC on Box%03d for %s"%(i, name)
        sampler.run_mcmc(pos,nsteps)
        print "MCMC complete for Box%03d\n"%(i)
        fullchain = sampler.flatchain
        likes = sampler.flatlnprobability
        chain = fullchain[nwalkers*nburn:]
        np.savetxt(base_dir+"chains/Box%03d_chain.txt"%(i),fullchain,header=header)
        np.savetxt(base_dir+"chains/Box%03d_likes.txt"%(i),likes)
        mean_models[i] = np.mean(chain,0)
        var_models[i] = np.var(chain,0)

    if run_mcmc_comparisons:
        for j in range(z_lo,z_hi):
            d,e,f,g = get_params(best_fit_models[i],scale_factors[j])
            TMF_array[j].set_parameters(d,e,f,g)
            N = TMF_array[j].n_in_bins(lM_bin_array[j])*volume
            N_err = np.sqrt(np.diagonal(cov_array[j]))
            sigdif = (N_data_array[j]-N)/N_err
            print "\nZ%d"%j
            for ind in range(len(N)):
                print "Bin %d: %.1f +- %.1f\tvs\t%.1f  at  %f"%(ind,N_data_array[j][ind],N_err[ind],N[ind],sigdif[ind])
            visualize.NM_plot(lM_array[j],N_data_array[j],N_err,lM_array[j],N,title="Box%03d at z=%.2f"%(i,redshifts[j]))

    if calculate_chi2:
        for j in range(z_lo,z_hi):
            d,e,f,g = get_params(best_fit_models[i],scale_factors[j])
            TMF_array[j].set_parameters(d,e,f,g)
            N_fit = TMF_array[j].n_in_bins(lM_bin_array[j])*volume
            N_data = N_data_array[j]
            X = N_data-N_fit
            cov = cov_array[j]
            icov = np.linalg.inv(cov)
            chi2 = np.dot(X,np.dot(icov,X))
            chi2s[i,j] = chi2
        print "Chi2s for Box%03d are:"%i
        print chi2s[i]

    #Save the models
    np.savetxt(base_save+"bests.txt",best_fit_models,header=header)
    np.savetxt(base_save+"means.txt",mean_models,header=header)
    np.savetxt(base_save+"vars.txt",var_models,header=header)
    np.savetxt(base_save+"BFchi2s.txt",chi2s)
    continue #end loop over boxes/cosmologies
