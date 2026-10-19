import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import hasasia.sensitivity as hasen
import hasasia.sim as hasim
from atnf_pull import psrcat
import hasasia.skymap as hasky
import healpy as hp
import astropy.units as u
import astropy.constants as c

DAY = 3600*24
YEAR = DAY*365 

# def compute_best_sens(config, Npsrs, cadence, freqs):

#     phi     = np.random.uniform(0, 2*np.pi,size=Npsrs) 
#     theta   = np.arccos(np.random.uniform(-1,1,size=Npsrs))
#     tspan   = [config["Tobs"] for ii in range(Npsrs)]

#     pta  = hasim.sim_pta(timespan=tspan, cad=cadence, sigma=config["err"], phi=phi, theta=theta, freqs=freqs)

    
#     spectra = []
#     for psr in pta:
#         sptrm = hasen.Spectrum(psr, freqs=freqs)
#         sptrm.NcalInv # computes inverse-noise-weighted transmission function for each pulsar
#         spectra.append(sptrm)

#     scGWB = hasen.GWBSensitivityCurve(spectra)


#     return np.min(scGWB.h_c)

# config = {
#     "Tobs"      : 5,
#     "err"       : 1e-7,
#     "nbins"     : 500,
#     "Npsr_max"  : 75,
#     "Ntoa_max"  : 2e3,
# }

# Npsr    = np.arange(2, config["Npsr_max"],2)
# Ntoa    = np.linspace(20, config["Ntoa_max"], 100)
# cad     = Ntoa // config["Tobs"]
# NPSR, CAD = np.meshgrid(Npsr, cad)
# freqs   = np.logspace(-10,-4,config["nbins"]) # frequency range [1/Tobs, nToAs/2Tobs]

# H = np.empty_like(NPSR, dtype=float)
# itmax = cad.size * Npsr.size

# for i in range(cad.size):
#     for j in range(Npsr.size):
#         h = compute_best_sens(config,Npsr[j],cad[i],freqs)
#         H[i,j] = h
#     if (i*j)%(itmax/10):
#         print(f"{i*j/itmax*100:.2f}% complete")
#     elif (i*j) == itmax-1:
#         print("done!")

f = np.load("/Users/consher/Downloads/pta_scaling.npz")
NPSR, CAD, H = f['NPSR'], f['CAD'], f['H']
print(NPSR,CAD,H)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(NPSR, CAD, H, vmin=H.min() * 2, cmap="Blues")
# ax.set_zscale('log')

# ax.set_xlabel('N_psr')
# ax.set_ylabel('cadence')
# ax.set_zlabel('h_min')


# plt.show()