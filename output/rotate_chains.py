"""
Take some results, such as efg, and rotate the chains to break tight correlations.
"""
import numpy as np
import corner, sys, os
import matplotlib.pyplot as plt

model = "dfgB"
#old_labels = [r"$e0$",r"$e1$",r"$f0$",r"$f1$",r"$g0$",r"$g1$"]

N_z     = 10
N_boxes = 39
N_p     = 2*len(model) #Number of parameters
mean_models = np.zeros((N_boxes, N_p))
var_models  = np.zeros((N_boxes, N_p))

#Just use Box000 to find the rotations
base_dir = "./%s"%model
inbase = base_dir+"/chains/Box%03d_chain.txt"
base_save = base_dir+"_rotated/"
chainout = base_save+"chains/rotated_Box%03d_chain.txt"
Rout     = base_save+"R_mats/R%03d.txt"

os.system("mkdir -p %s"%base_save)
os.system("mkdir -p %s/chains"%base_save)
os.system("mkdir -p %s/R_mats"%base_save)


index = 0
inbase = base_dir+"/chains/Box%03d_chain.txt"

make_Rs = False
rotate = True

if make_Rs:
    #First find all the rotation matrices
    for i in range(0,N_boxes):
        data = np.loadtxt(inbase%i)
        D = np.copy(data)
        C = np.cov(D,rowvar=False)
        w,R = np.linalg.eig(C)
        np.savetxt(Rout%i,R)
        #As it turns out, cosmo 34 is the middle-most box,
        #so use it for the final rotation matrix.
        if i == 34: np.savetxt(base_save+"/R_matrix.txt",R)
        print "Created R%d"%i

if rotate:
    #First get the Rotation matrix
    R = np.loadtxt(base_save+"/R_matrix.txt")
    #Now rotate some chains
    for i in range(0,N_boxes):
        data = np.loadtxt(inbase%i)
        imeans = np.mean(data,0)
        rD = np.dot(data[:],R) #Rotated data
        np.savetxt(chainout%i,rD)
        mean_models[i] = np.mean(rD,0)
        var_models[i] = np.var(rD,0)
        print "Saved box%03d"%i
        #fig = corner.corner(data,labels=old_labels)
        #fig = corner.corner(rD)
        #plt.show()
        #if i ==2:
        #    sys.exit()
    np.savetxt(base_save+"/rotated_%s_means.txt"%model, mean_models)
    np.savetxt(base_save+"/rotated_%s_vars.txt"%model,  var_models)

