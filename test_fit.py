import clusterwl
import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt

cosmologies = np.genfromtxt("cosmos.txt")
num,ombh2,omch2,w0,n_s,ln10As,H0,Neff,sigma8 = cosmologies[0]
h = H0/100.
Oc = omch2/h**2
Ob = ombh2/h**2
Om = Oc+Ob
A_s = np.exp(ln10As)/10e10
p = ccl.Parameters(Omega_c=Oc, Omega_b=Ob, h=h, A_s=A_s, n_s=n_s)
cosmo = ccl.Cosmology(p)
k = np.logspace(-5, 2, base=10, num=1000) #h/Mpc

sfs = np.array([0.25, 0.333333, 0.5, 0.540541, 0.588235, 
                0.645161, 0.714286, 0.8, 0.909091, 1.0])
volume = 1050.**3 #Mpc^3/h^3
NM = 1000
N_z = len(sfs)

base = "../../all_MF_data/building_MF_data/"#"../Mass-Function-Emulator/test_data/"
datapath = base+"N_data/Box%03d_full/Box%03d_full_Z%d.txt"
covpath  = base+"covariances/Box%03d_cov/Box%03d_cov_Z%d.txt"
def get_basepaths():
    return [base, datapath, covpath]
def get_testbox_data(sim_index, z_index):
    base, datapath, covpath = get_basepaths()
    data = np.loadtxt(datapath%(sim_index, sim_index, z_index))
    N = data[:,2]
    goodinds = N>0
    data = data[goodinds]
    lM_bins = data[:,:2]
    lM = np.mean(lM_bins, 1)
    N = data[:,2]
    cov = np.loadtxt(covpath%(sim_index, sim_index, z_index))
    cov = cov[goodinds]
    cov = cov[:,goodinds]
    err = np.sqrt(np.diagonal(cov))
    return lM_bins, lM, N, err, cov


def get_colors(cmapstring="seismic"):
    cmap = plt.get_cmap(cmapstring)
    return [cmap(ci) for ci in np.linspace(1.0, 0.0, N_z)]
colors = get_colors()

print "Making P(k)"
Ps = [ccl.linear_matter_power(cosmo, k*h, a)*h**3  for a in sfs]
print "Done with P(k)"

def plot_N():
    Marr = np.logspace(12, 14.1, NM, base=10)
    dndM = np.zeros_like(Marr)

    for i in range(len(sfs)):
        clusterwl.massfunction.calc_dndM_at_M(Marr, k, Ps[i], Om, dndM)
        edges = np.logspace(np.log10(min(Marr)), np.log10(max(Marr)), 11, base=10)
        N = clusterwl.massfunction.N_in_bins(edges, volume, Marr, dndM)
        M = (edges[:-1]+edges[1:])/2.
        plt.loglog(M, N, label=r"$a=%.2f$"%sfs[i], c=colors[i])
        lMb, lM, Ndata, err, cov = get_testbox_data(0,i)
        plt.errorbar(10**lM, Ndata, err, c=colors[i])
        print "done with a=%.2f"%sfs[i]
    plt.legend()
    plt.ylim(1e-1, 1e7)
    plt.show()

if __name__ == "__main__":
    plot_N()
