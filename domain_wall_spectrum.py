import json
import numpy as np
import pandas as pd

HUBBLE = 2.20e-18 # Hubble constant (68km/s/Mpc)
H = 0.6766 # the reduced hubble constant H0/100

def main():
    generate_specta("./data/new-physics_data/domain_walls.json",1000)

def generate_specta(config, nbins):
    '''
    generate a domain wall spectrum assuming Omega~f^3 for f<f_peak and Omega~f^-1 for f>f_peak
    
    config : config file of dw parameters
    nbins : number of frequency bins to compute for
    '''

    # read config file
    with open(config, 'r') as fp:
        dw = json.load(fp)

    # get parameters from dictionary
    sigma = dw['sigma']             # domain wall tension in TeV^3
    T_ann = dw['T_ann']             # domain wall annihilation tempeature in GeV
    g_energy = dw['g_energy']       # relativistic energy degrees of freedom at temp T_ann
    g_entropy = dw['g_entropy']     # relativistic entropy degrees of freedom at temp T_ann
    A = dw['area_param']            # area parameter estimated as 0.8 +- 0.1
    epsilon_gw = dw['efficiency_param']   # efficiency parameter estimated to be 0.7 +- 0.4

    # compute the peak frequency and the value of Omega_gw at that frequency
    f_peak = 1.1e-9 * (g_energy / 10)**(1/2.) * (g_entropy / 10)**(-1/3.) * (T_ann / 1e-2)
    Omega_peak = ( 7.2e-18 * epsilon_gw * A**2 * (g_energy / 10)**(-4/3.) * (sigma / 1)**2 * (T_ann/ 1e-2)**(-4) ) / (H**2)

    # create frequency and energy density arrays
    f = np.logspace(-12, 3, nbins)
    Omega_gw = np.empty(nbins)

    for i in range(0,nbins):
        if f[i] <= f_peak:
            Omega_gw[i] = Omega_peak * (f[i] / f_peak)**3
        else:
            Omega_gw[i] = Omega_peak * (f[i] / f_peak)**(-1)

    # write output
    output = pd.DataFrame({'f': f, 'Omega_gw': Omega_gw})
    output.to_csv("./data/new-physics_data/dw_spectrum.dat",index=False,sep=' ')

if __name__ =="__main__":
    main()