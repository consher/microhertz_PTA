# author: Conor Sheridan
# date: 2026/10/12
# description: plotting code for paper

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import healpy as hp
import json
from astropy import units as u
from astropy.coordinates import SkyCoord

HUBBLE = 2.20e-18 # Hubble constant (68km/s/Mpc)
H = 0.6766 # the reduced hubble constant H0/100
DAY = 24*3600
YEAR = 365*DAY

# -------------
def main():
    # run desired plotting functions here
    plot_dir = "./plots"
    data_dir = "./output_data"
    plot_skymap_SNR(data_dir, plot_dir,'mupta_skymap_omega_twin-campaign.pdf',plot_freq=1e-6, strain=8.9e-14)
    plot_plsc_EDensity(data_dir, plot_dir,"mupta_plsc_omega_twin-campaign.pdf")
    dsa_ska_plot(data_dir, plot_dir, "dsa-ska_plcurve-skymap.pdf",4.9e-14, check_freq=False)
# -------------

def h_to_omega(f,h_c):
    '''
    vectorised function for converting characterstic strain to fractional energy density
    
    f : frequency array
    h_c : the characteristic strain values to convert to fractional energy density values
    '''
    def eqn(f,h_c):
        return (2 * np.pi**2) / (3 * HUBBLE**2) * (f**2) * (h_c**2)
    vec_eqn=np.vectorize(eqn) # convert function to vector function 
    return vec_eqn(f,h_c)

def omega_to_h(f,omega_gw):
    '''
    vectorised function for converting characterstic strain to fractional energy density
    
    f : frequency array
    omega_gw : the fractional energy density values to convert to characteristic strain
    '''
    def eqn(f,omega_gw):
        return np.sqrt((2/3.)*omega_gw) * (HUBBLE/np.pi) * (1/f)  
    vec_eqn=np.vectorize(eqn) # convert function to vector function 
    return vec_eqn(f,omega_gw)

def S_to_omega(f,S_eff):
    '''
    vectorised function for converting power density to fractional energy density
    
    f : frequency array
    S_eff : the effective noise power density to convert to fractional energy density values
    '''
    def eqn(f,S_eff):
        return (2 * np.pi**2) / (3 * HUBBLE**2) * (f**3) * S_eff
    vec_eqn=np.vectorize(eqn) # convert function to vector function 
    return vec_eqn(f,S_eff)


def plot_dw_spectra(ax, nbins, fill=False, color="#1D5C28"):
    '''
    plot domain wall GW signature, with Omega~f^3 for f<f_peak and Omega~f^-1 for f>f_peak from .dat file

    ax      : axis to plot data on
    nbins   : number of frequency bins
    fill    : shade area below curve
    color   : colour of curve
    '''

    # read data
    dw_data = np.asarray(pd.read_csv("data/new-physics_data/dw_spectrum.dat", delimiter=' ', header=0))

    # plot data (omega_gw vs freq)
    ax.loglog(dw_data[:,0], dw_data[:,1], label="Domain Walls", linestyle='-.', color=color)

    # fill below curve if requested
    if fill == True:
        lower = np.zeros(nbins) + 1e-30
        ax.fill_between(dw_data[:,0], dw_data[:,1], lower, color='#1D5C28', alpha=0.2)


def plot_fopt_spectra(ax, T, plot_individually=False, fill=False, color="#1C2320"):
    '''plot First Order Phase Transitions from (Renzini et al. 2022) for a 200MeV & 200GeV transition respectively
    
    ax                  : axis to plot on, use None if plotting individually
    T                   : List of the energies to plot, here 200 MeV or 200 GeV
    plot_individually   : plot data on it's own figure
    fill                : fill below curve with shading
    color               : colour of curve
    '''

    fotp_200MeV = np.asarray(pd.read_csv("data/new-physics_data/fotp200MeV_spectrum.dat", delimiter=' ', header=0))
    fotp_200GeV = np.asarray(pd.read_csv("data/new-physics_data/fotp200GeV_spectrum.dat", delimiter=' ', header=0))

    # set up axes if for individual plotting
    if plot_individually == True:
        # plt.style.use(['science','ieee','no-latex'])
        # plt.rcParams['font.size'] = 12
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Frequency $f_\\text{gw}$ [Hz]',)
        ax.set_ylabel('Characteristic Strain $h_c$')

    lower = np.zeros(len(fotp_200GeV)) + 1e-30 # lower bound for ax.fill_between()

    if '200GeV' in T:
        ax.plot(fotp_200GeV[:,0], fotp_200GeV[:,1], label="Phase Transition", color=color)
        if fill == True:
            ax.fill_between(fotp_200GeV[:,0], fotp_200GeV[:,1], lower, color=color, alpha=0.2)

    if '200MeV' in T:
        ax.plot(fotp_200MeV[:,0], fotp_200MeV[:,1], label="Phase Transition", linestyle='--', color=color)  
        if fill == True:
            ax.fill_between(fotp_200MeV[:,0], fotp_200MeV[:,1], lower, color=color, alpha=0.2)



def plot_cs_spectra(ax, Gmu_list,units='density', fill=True, plot_individually=False, color='#000000'):
    '''Plot cosmic string spectra from (Jose J. Blanco-Pillado, Ken D. Olum 2017) for list of tensions (in the form integer orders of magnitude (e.g Gmu=1e-10 --> [-10]))
    
    ax                  : axis to plot on, use None if plotting individually
    Gmu_list            : list of desired cosmic string tensions to plot from data e.g. [-9,-10] corresponds to plotting the 1e-9 and 1e-10 curves.
    units               : desired units to plot in, either 'density' or 'strain'. Default='density'
    fill                : fill space between curves with shading
    plot_individually   : plot data on it's own figure
    color               : colour of curve/fill
    '''
    # cosmic string values
    cs_spectra = pd.read_csv("data/new-physics_data/cs_spectrum.dat", delimiter=' ', header=0)
    cs_spectra = np.asarray(cs_spectra)

    # number of datapoints per tension
    nfreq = 2501

    # get frequency array
    csfreqs = cs_spectra[0:nfreq,1]

    # set up axes if for individual plotting
    if plot_individually == True:
        plt.style.use(['science','ieee','no-latex'])
        plt.rcParams['font.size'] = 12
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(1,1,1)

    count = 0
    for Gmu in Gmu_list:
        Gmu_ = (-1*Gmu - 8)*4 # how integer values of Gmu are stored in the data
        omega_cs = cs_spectra[nfreq*Gmu_:nfreq*(Gmu_+1),2] / (H**2) # pull h^2Omega_gw from array and divide by reduced hubble squared

        lower = np.zeros(nfreq) + 1e-30 # default lower bound for ax.fill_between()

        if fill == True and count == 0:
            upper = omega_cs
            if units == 'strain':
                upper = omega_to_h(csfreqs, upper)
        elif fill == True and count != 0:
            lower = omega_cs
            if units == 'strain':
                lower = omega_to_h(csfreqs, lower)

        if units == 'strain':
            # convert from omega (energy density) to strain
            h_cs = np.sqrt((2/3.)*omega_cs) * (HUBBLE/np.pi) * (1/csfreqs)
            if count == 0:
                ax.loglog(csfreqs, h_cs, color=color, linestyle='-', lw=0.8, label="Cosmic Strings")
            else:
                ax.loglog(csfreqs, h_cs, color=color, linestyle='-', lw=0.8)
        elif units == 'density':
            # otherwise plot density
            if count == 0:
                ax.loglog(csfreqs, omega_cs, color=color, linestyle='-', lw=0.8, label="Cosmic Strings")
            else:
                ax.loglog(csfreqs, omega_cs, color=color, linestyle='-', lw=0.8)
        else:
            print("\n\nNot a valid unit!\n\n")
            raise ValueError

    if fill == True:
        ax.fill_between(csfreqs, upper, lower, color=color, alpha=0.2)

    if plot_individually == True:
        fmin = 2e-9
        fmax = 1e-6
        ymin = 1e-16
        ymax = 1e-13
        plt.xlim(fmin, fmax)
        plt.ylim(ymin, ymax)
        ax.set_xlabel('Frequency $f_\\text{gw}$ [Hz]',)
        ax.set_ylabel('Characteristic Strain $h_c$')

def plot_plsc_EDensity(data_dir,plot_dir,NAME):
        '''
        Plots power-law sensitivity curves, and GW targets if interest, in units of energy density (Omega_gw)
        
        plot_dir    : plot output
        NAME        : name of output plot (excluding .pdf suffix)
        '''

        # plotting style
        plt.style.use(['science','ieee','no-latex'])
        plt.rcParams['font.size'] = 12
        fig = plt.figure(1, figsize=(6,8))
        axes = fig.subplots(2,1)
    
        # plotting options
        fmin = 1e-9
        fmax = 1e-3
        ymin = 1e-15
        ymax = 1e6
    
        # frequency and strain are the first and second indices respectively
        mupta_plcurve_wsrtA = np.loadtxt(data_dir+"/mupta_plcurve_2-1_wsrt.csv", delimiter=",")
        mupta_plcurve_fastA = np.loadtxt(data_dir+"/mupta_plcurve_2-1_fast.csv", delimiter=",")
        mupta_plcurve_wsrtB = np.loadtxt(data_dir+"/mupta_plcurve_4-2_wsrt.csv", delimiter=",")
        mupta_plcurve_fastB = np.loadtxt(data_dir+"/mupta_plcurve_4-2_fast.csv", delimiter=",")
        epta_scurve  = np.loadtxt(data_dir+"/epta_plcurve.csv", delimiter=",")
    
        # convert from characteristic strain to omega
        mupta_plcurve_wsrtA = [mupta_plcurve_wsrtA[0], h_to_omega(mupta_plcurve_wsrtA[0],mupta_plcurve_wsrtA[1]), h_to_omega(mupta_plcurve_wsrtA[0],mupta_plcurve_wsrtA[2]), h_to_omega(mupta_plcurve_wsrtA[0],mupta_plcurve_wsrtA[3])]
        mupta_plcurve_fastA = [mupta_plcurve_fastA[0], h_to_omega(mupta_plcurve_fastA[0],mupta_plcurve_fastA[1]), h_to_omega(mupta_plcurve_fastA[0],mupta_plcurve_fastA[2]), h_to_omega(mupta_plcurve_fastA[0],mupta_plcurve_fastA[3])]
        epta_scurve       = [epta_scurve[0], h_to_omega(epta_scurve[0],epta_scurve[1]), h_to_omega(epta_scurve[0],epta_scurve[2]), h_to_omega(epta_scurve[0],epta_scurve[3])]
        mupta_plcurve_wsrtB = [mupta_plcurve_wsrtB[0], h_to_omega(mupta_plcurve_wsrtB[0],mupta_plcurve_wsrtB[1]), h_to_omega(mupta_plcurve_wsrtB[0],mupta_plcurve_wsrtB[2]), h_to_omega(mupta_plcurve_wsrtB[0],mupta_plcurve_wsrtB[3])]
        mupta_plcurve_fastB = [mupta_plcurve_fastB[0], h_to_omega(mupta_plcurve_fastB[0],mupta_plcurve_fastB[1]), h_to_omega(mupta_plcurve_fastB[0],mupta_plcurve_fastB[2]), h_to_omega(mupta_plcurve_fastB[0],mupta_plcurve_fastB[3])]

        print("\nTelescope\tCampaign\tMin Omega_gw")
        print(f"WSRT\t\tA\t\t{np.min(mupta_plcurve_wsrtA[2])}")
        print(f"WSRT\t\tB\t\t{np.min(mupta_plcurve_wsrtB[2])}")
        print(f"FAST\t\tA\t\t{np.min(mupta_plcurve_fastA[2])}")
        print(f"FAST\t\tB\t\t{np.min(mupta_plcurve_fastB[2])}\n")

        # plots for Campaign A (2 TOAs per PSR, T_obs = 1yr)
        axes[0].loglog(mupta_plcurve_wsrtA[0], mupta_plcurve_wsrtA[2],linestyle='-',color="#73ab57",lw=1.5) # SNR = 5 plot of WSRT
        axes[0].loglog(mupta_plcurve_fastA[0], mupta_plcurve_fastA[2],linestyle='-',color="#B50707",lw=1.5) # SNR = 5 plot of FAST

        # plots for Campaign B (4 TOAs per PSR, T_obs = 2yr)
        axes[1].loglog(mupta_plcurve_wsrtB[0], mupta_plcurve_wsrtB[2],linestyle='-',color='#73ab57',lw=1.5) # SNR = 5 plot of WSRT
        axes[1].loglog(mupta_plcurve_fastB[0], mupta_plcurve_fastB[2],linestyle='-',color="#B50707",lw=1.5) # SNR = 5 plot of FAST

        # apply to both plots
        for ax in axes:
            # plot New-Physics targets
            plot_fopt_spectra(ax, T=['200MeV'], fill=True, color="#1C2320") # plot first order phase transition
            ax.annotate(("$T_*\\sim 200$ MeV"), xy=(1e-7,5e-13), color="#1C2320")
            plot_cs_spectra(ax, [-10], fill=True, color="#606060") # plot cosmic string background spectra
            ax.annotate(("$G\\mu\\sim10^{-10}$"), xy=(5e-5,5e-9), color="#606060")
            plot_dw_spectra(ax, nbins=1000, fill=True, color="#1D5C28") # plot domain wall spectrum with 1000 frequency bins
            ax.annotate("$T_\\text{ann}\\sim1$ GeV", xy=(4e-5,1e-13), color='#1D5C28')

            ax.text(mupta_plcurve_wsrtB[0][-220], mupta_plcurve_wsrtB[2][-100], "WSRT", color="#73ab57")
            ax.text(mupta_plcurve_fastB[0][-25], mupta_plcurve_fastB[2][-80], "FAST", color="#B50707")

            ax.loglog(epta_scurve[0], epta_scurve[2],linestyle='-',color="#9437db",lw=1.5,label='EPTA Benchmark') # SNR = 5 plot of EPTA DR2
        
            # misc axis settings
            ax.set_xlabel('Frequency $f_\\text{gw}$ [Hz]',)
            ax.set_ylabel('Energy Density $\\Omega_\\text{gw}$')
            ax.set_xlim(fmin,fmax)
            ax.set_ylim(ymin,ymax)
            # rearrange legend items to new order for aesthetic purposesß
            handles, labels = ax.get_legend_handles_labels() 
            order = [3,0,1,2]   
            ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left')

        plt.savefig(f"{plot_dir}/{NAME}")

def plot_skymap_SNR(data_dir,plot_dir,NAME,plot_freq,strain):
    '''
    Plots skymap singal-to-noise ratio for a given strain
    
    plot_dir    : plot output
    NAME        : name of output plot (excluding .pdf suffix)
    plot_freq   : GW frequency to plot
    strain      : strain of source to calculate SNR of
    '''

    # plotting style
    fig = plt.figure(2,figsize=(20,10))
    plt.style.use(['science','ieee','no-latex'])
    plt.rcParams['font.size'] = 20

    # load data
    skymap_wsrt_A = np.loadtxt(data_dir+"/mupta_skymap_2-1_wsrt.csv", delimiter=",")
    skymap_fast_A = np.loadtxt(data_dir+"/mupta_skymap_2-1_fast.csv", delimiter=",")
    skymap_wsrt_B = np.loadtxt(data_dir+"/mupta_skymap_4-2_wsrt.csv", delimiter=",")
    skymap_fast_B = np.loadtxt(data_dir+"/mupta_skymap_4-2_fast.csv", delimiter=",")
    psr_coords  = np.loadtxt(data_dir+"/mupta_psr_coords.csv", delimiter=",")
    sky_coords  = np.loadtxt(data_dir+"/mupta_sky_coords.csv", delimiter=",")
    #freqs  = np.loadtxt(data_dir+"/freqrange.csv", delimiter=",")

    # calculate index of skymap matching desired frequency; 20=1e-8, 30=1e-7, 40=1e-6,... 
    skymap_freq_idx = int(10 * (10 + np.log10(plot_freq)))

    npsr = len(psr_coords[0,:]) # get number of pulsars

    # format the pulsar coordinates to show correctly on the plot
    for i in range(npsr):
        if(psr_coords[0,i] >= 0):
            psr_coords[0,i] = np.pi/2. - abs(psr_coords[0,i]) # add latitude to pi/2 to be below ecliptic
        elif(psr_coords[0,i] < 0):
            psr_coords[0,i] = np.pi/2. + abs(psr_coords[0,i]) # subtract latitude to pi/2 to be below ecliptic
        if(psr_coords[1,i] > np.pi and psr_coords[1,i] < 2*np.pi):
            psr_coords[1,i] -= 2*np.pi # subtract 2*pi from any angle in the range [pi, 2*pi] to keep in bounds

    # get the best sensitivity for all skymaps in units of S_eff
    wsrt_min_A = np.min(skymap_wsrt_A[skymap_freq_idx])
    fast_min_A = np.min(skymap_fast_A[skymap_freq_idx])
    wsrt_min_B = np.min(skymap_wsrt_B[skymap_freq_idx])
    fast_min_B = np.min(skymap_fast_B[skymap_freq_idx])

    # get optimal position (will be the same for all PTA configurations as pulsar choice doesn't change, so get optimal position of just one config)
    optimal_idx = np.where(skymap_fast_B[skymap_freq_idx] == fast_min_B)[0][0]
    theta_min = sky_coords[0][optimal_idx]
    phi_min = sky_coords[1][optimal_idx]

    if(theta_min >= 0):
        theta_min = np.pi/2. - abs(theta_min) # add latitude to pi/2 to be below ecliptic
    elif(theta_min < 0):
        theta_min = np.pi/2. + abs(theta_min) # subtract latitude to pi/2 to be below ecliptic
    if(phi_min > np.pi and phi_min < 2*np.pi):
        phi_min -= 2*np.pi # subtract 2*pi from any angle in the range [pi, 2*pi] to keep in bounds

    # Transform to equatorial coordinates
    theta_min_deg   = np.rad2deg(theta_min)
    phi_min_deg     = np.rad2deg(phi_min)
    ecliptic_coord = SkyCoord(lat=theta_min_deg*u.degree, lon=phi_min_deg*u.degree, frame='barycentrictrueecliptic')
    equatorial_coord = ecliptic_coord.transform_to('icrs')

    # print the direction of best-sensitivity
    print("\nSky-Map Optimal Direction:")
    print('Ecliptic\t {0:2.1f} degrees N, {1:2.1f} degrees E'.format(theta_min_deg,phi_min_deg))
    print('Equatorial\t RA {0:2.1f}, DEC {1:2.1f}'.format(equatorial_coord.ra,equatorial_coord.dec))

    # compute snr for a given strain
    T_obsA = 1*YEAR
    T_obsB = 2*YEAR
    snr_wsrt_A = strain * np.sqrt(T_obsA/skymap_wsrt_A[skymap_freq_idx])
    snr_wsrt_B = strain * np.sqrt(T_obsB/skymap_wsrt_B[skymap_freq_idx])
    snr_fast_A = strain * np.sqrt(T_obsA/skymap_fast_A[skymap_freq_idx])
    snr_fast_B = strain * np.sqrt(T_obsB/skymap_fast_B[skymap_freq_idx])

    # plot molleide projection of for each campaign A & B and for each telescope WSRT & FAST
    sub = 221
    skymap_snr_ar = [snr_wsrt_A,snr_wsrt_B,snr_fast_A,snr_fast_B]
    skymap_title_ar = ["WSRT Campaign A", "WSRT Campaign B", "FAST Campaign A", "FAST Campaign B"]
    cbar_tick_ar = [[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5]]

    for i in range(0,4):
        hp.projview(
            skymap_snr_ar[i],
            fig=2,
            sub=sub,
            projection_type="mollweide",
            coord='E',
            #graticule_coord='E',
            cmap="plasma",
            unit="$\\rho$",
            norm="linear",
            title=skymap_title_ar[i]+"\n$\\rho_\\text{max}=$"+f"{np.max(skymap_snr_ar[i]):.2f}", 
            cbar=True,
            cb_orientation="vertical",
            show_tickmarkers=False,
            cbar_ticks=cbar_tick_ar[i],
            graticule=True,
            graticule_labels=True,
            graticule_color='k',
            longitude_grid_spacing=30, 
            latitude_grid_spacing=30,
            custom_xtick_labels=[f"{int(-1*i)}$^\\circ$" for i in np.linspace(-150,150,11)],
            custom_ytick_labels=[f"{int(i)}$^\\circ$" for i in np.linspace(-60,60,5)],
            flip="astro",
            fontsize={"cbar_label":18, "cbar_tick_label":16}
        )

        # overplot pulsar locations using modified psr_coords
        hp.newprojplot(psr_coords[0,:], psr_coords[1,:], "w*", markersize=12, markeredgecolor="k")

        sub += 1 # update the subplot to the next one
    # --------------

    plt.savefig(f"{plot_dir}/{NAME}")

def dsa_ska_plot(data_dir,plot_dir,NAME, strain, check_freq=False):
    '''
        Plots power-law sensitivity curve and skymap of DSA + SKA-Mid PTA
        
        plot_dir    : plot output
        NAME        : name of output plot (excluding .pdf suffix)
        strain      : desired strain to calculate SNR of for skymap
        check_freq  : check the frequency you are plotting on skymap
        '''

    fig = plt.figure(3,figsize=(16,5))
    plt.style.use(['science','ieee','no-latex'])
    plt.rcParams['font.size'] = 20
    axes = fig.subplots(1,2)

    # plotting options
    fmin = 1e-9
    fmax = 1e-3
    ymin = 1e-15
    ymax = 1e6

    # number of pulsars
    npsr = 24
    Tobs = 2

    # load data
    mupta_plcurve_fastB     = np.loadtxt(data_dir+"/mupta_plcurve_4-2_fast.csv", delimiter=",")
    freqs, dsa_ska_plcurve  = np.loadtxt(data_dir+"/dsa-ska_plcurve.csv", delimiter=",")
    dsa_ska_skymap          = np.loadtxt(data_dir+"/dsa-ska_skymap.csv", delimiter=",")
    psr_coords              = np.loadtxt(data_dir+"/dsa-ska_psr_coords.csv", delimiter=",")

    # plot the FAST campaign B and DSA+SKA PL curve
    axes[0].loglog(freqs, dsa_ska_plcurve,linestyle='-',color="#EF78EF",lw=1.5,label="DSA + SKA-Mid") 
    axes[0].loglog(mupta_plcurve_fastB[0], h_to_omega(mupta_plcurve_fastB[0],mupta_plcurve_fastB[2]),linestyle='-',color="#B50707",lw=1.5, label="FAST Campaign B")


    # plot New-Physics targets
    plot_fopt_spectra(axes[0], T=['200MeV'], fill=True, color="#1C2320") # plot first order phase transition
    axes[0].annotate(("$T_*\\sim 200$ MeV"), xy=(1e-7,5e-13), color="#1C2320")
    plot_cs_spectra(axes[0], [-10], fill=True, color="#606060") # plot cosmic string background spectra
    axes[0].annotate(("$G\\mu\\sim10^{-10}$"), xy=(5e-5,5e-9), color="#606060")
    plot_dw_spectra(axes[0], nbins=1000, fill=True, color="#1D5C28") # plot domain wall spectrum with 1000 frequency bins
    axes[0].annotate("$T_\\text{ann}\\sim1$ GeV", xy=(4e-5,1e-13), color='#1D5C28')

    # misc axis settings
    axes[0].set_xlabel('Frequency $f_\\text{gw}$ [Hz]',)
    axes[0].set_ylabel('Energy Density $\\Omega_\\text{gw}$')
    axes[0].set_xlim(fmin,fmax)
    axes[0].set_ylim(ymin,ymax)
    # rearrange legend items to new order for aesthetic purposesß
    handles, labels = axes[0].get_legend_handles_labels() 
    order = [0,1,2,3,4]   
    axes[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left')

    axes[1].axis('off') # turn off square axes

    # get desired frequency to plot skymap of and optionally check
    fidx = 351
    if check_freq == True:
        print(f"Skymap Freq = {freqs[fidx]:.2e} Hz")

    snr = strain * np.sqrt(Tobs*YEAR/dsa_ska_skymap[fidx]) # compute SNR for given strain

    cbar_ticks = [0,1,2,3,4,5,6,7,8,9,10]
    hp.projview(
        snr,
        fig=3,
        sub=122,
        projection_type="mollweide",
        coord='E',
        #graticule_coord='E',
        cmap="plasma",
        unit="$\\rho$",
        norm="linear",
        title="DSA + SKA-Mid\n$\\rho_\\text{max}=$"+f"{np.max(snr):.2f}", 
        cbar=True,
        cb_orientation="vertical",
        show_tickmarkers=False,
        cbar_ticks=cbar_ticks,
        graticule=True,
        graticule_labels=True,
        graticule_color='k',
        longitude_grid_spacing=30, 
        latitude_grid_spacing=30,
        custom_xtick_labels=[f"{int(-1*i)}$^\\circ$" for i in np.linspace(-150,150,11)],
        custom_ytick_labels=[f"{int(i)}$^\\circ$" for i in np.linspace(-60,60,5)],
        flip="astro",
        fontsize={"cbar_label":18, "cbar_tick_label":16}
    )

    # format the pulsar coordinates to show correctly on the plot
    for i in range(npsr):
        if(psr_coords[0,i] >= 0):
            psr_coords[0,i] = np.pi/2. - abs(psr_coords[0,i]) # add latitude to pi/2 to be below ecliptic
        elif(psr_coords[0,i] < 0):
            psr_coords[0,i] = np.pi/2. + abs(psr_coords[0,i]) # subtract latitude to pi/2 to be below ecliptic
        if(psr_coords[1,i] > np.pi and psr_coords[1,i] < 2*np.pi):
            psr_coords[1,i] -= 2*np.pi # subtract 2*pi from any angle in the range [pi, 2*pi] to keep in bounds

    # overplot pulsar locations using modified psr_coords
    hp.newprojplot(psr_coords[0,:], psr_coords[1,:], "w*", markersize=12, markeredgecolor="k")

    plt.savefig(f"{plot_dir}/{NAME}")


if __name__ == "__main__":
    main()