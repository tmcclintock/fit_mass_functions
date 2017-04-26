"""
Create various chi2 histogram plots.
"""
import numpy as np
import sys
from scipy.stats import chi2
import matplotlib.pyplot as plt

base_dir = "output/efg/"
base_save = base_dir+"efg_"
chi2s = np.loadtxt(base_save+"BFchi2s.txt")
df = 10.0 #approximately

fchi2s = chi2s.flatten()
plt.hist(fchi2s, 40, normed=True)
x = np.linspace(chi2.ppf(0.01, df), chi2.ppf(0.99, df), 100)
plt.plot(x, chi2.pdf(x, df))
plt.xlabel(r"$\chi^2$",fontsize=24)
plt.title(r"Best fit with $[e,\ f,\ g]$")
plt.subplots_adjust(bottom=0.15)
plt.show()
