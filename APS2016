import numpy as np
import matplotlib as plt
import math
import kwant
import scipy.linalg as la
"""Kwant tutorial on discretizing Hamiltonian and computing the spectrum"""
# Particle in 2D yielding a square lattice with lattice constant a set to the value of 1
lat = kwant.lattice.square(a=1)
# Confine electrons in a finite circle region
t = 1 # energy unit
e = 15 # eccentricity
r1=2 # axis 1
r2=3 # axis 2
# define ellipse 
def ellipse(pos):
    x, y = pos
    return (x/r1)**2 + (y/r2)**2 < e**2


sys = kwant.Builder() # build system
sys[lat.shape(ellipse, (0,0))] = 4 * t # onsite energy for 2D is 4t, shape is obviously ellipse and 0,0 the origin for x and y
sys[lat.neighbors()] = -t # hopping between nearest neighbours is -t
sys = sys.finalized() # finalize the system so that one can perform calculations with it
# need to define the Hamiltonian of the system
ham = sys.hamiltonian_submatrix()
# do something like compute eigenvalues and eigenfunctions of the system
eval, evec = la.eigh(ham)
kwant.plotter.map(sys, abs(evec[:,0])**2)
# Make it more interactive: show first 10 wavefunctions
from ipywidgets import interact # Creates user interface (UI) controls for interactive use code and data

def plot_wf(i=0):
    print("Wavefunction with index", i)
    print("Energy:", eval[i],"t") 
    kwant.plotter.map(sys, abs(evec[:, i])**2)
    
interact(plot_wf, i=(0, 10))