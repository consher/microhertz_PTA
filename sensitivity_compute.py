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

HUBBLE = 67.66*(1e3/3.0857e+22) # Hubble constant (from Planck 2018)
H = 0.6766 # the reduced hubble constant H0/100
DAY = 24*3600
YEAR = 365*DAY


def main():
    # CONFIGURABLE PARAMETERS (make separate config file)
    # PSR catalog for ephemerides
    catalog = psrcat(".")
    config_file = "./pta_config.json"
    observatory_params = "./data/observatories/"

    # naming convention
    scurve_filename = f"mupta_scurve_{cadence}-{Tobs}" # output namestem for the sensitivity curves
    skymap_filename = f"mupta_skymap_{cadence}-{Tobs}" # output namestem for the sky maps


    with open("./pta_config.json", 'r') as fp:
        pta_config = json.load(fp)

    psr_list = pta_config["psr_list"] # list of pulsars in array
    Tobs = pta_config["Tobs"]         # years, total observation time
    cadence = pta_config["cadence"]   # cadence in number per day
    toa_int = pta_config["toa_int"]   # seconds, integration time to create one ToA

    # get observatory parameters
    with open(observatory_params+"wsrt_params.json", 'r') as fp:
        wsrt_params = json.load(fp)
    with open(observatory_params+"fast_params.json", 'r') as fp:
        fast_params = json.load(fp)

    # compute rms error array for WSRT and FAST
    rmserr_wsrt = np.array([rms_error(wsrt_params, catalog.get(psr), toa_int) for psr in psr_list], dtype=float)
    rmserr_fast = np.array([rms_error(fast_params, catalog.get(psr), toa_int) for psr in psr_list], dtype=float)

    #  -- WSRT --
    # scurve opject has frequency in first index and SNR 1, 10 sensitivity curve in second and third index respectively
    freqs, scurve1  = gwb_PLSC(psr_list=psr_list, Tobs=Tobs, cadence=cadence, snr=1, rmserr=rmserr_wsrt)
    freqs, scurve5  = gwb_PLSC(psr_list=psr_list, Tobs=Tobs, cadence=cadence, snr=5, rmserr=rmserr_wsrt)
    freqs, scurve10 = gwb_PLSC(psr_list=psr_list, Tobs=Tobs, cadence=cadence, snr=10, rmserr=rmserr_wsrt)
    senscurve = np.asarray([freqs, scurve1, scurve5, scurve10])

    # save to csv file for plotting
    np.savetxt("data/output_data/"+scurve_filename+"_wsrt.csv", senscurve, delimiter=",")

    # -- FAST --
    # scurve opject has frequency in first index and SNR 1, 5, 10 sensitivity curve in second and third index respectively
    freqs, scurve1  = gwb_PLSC(psr_list=psr_list, Tobs=Tobs, cadence=cadence, snr=1, rmserr=rmserr_fast)
    freqs, scurve5  = gwb_PLSC(psr_list=psr_list, Tobs=Tobs, cadence=cadence, snr=5, rmserr=rmserr_fast)
    freqs, scurve10 = gwb_PLSC(psr_list=psr_list, Tobs=Tobs, cadence=cadence, snr=10, rmserr=rmserr_fast)
    senscurve = np.asarray([freqs, scurve1, scurve5, scurve10])

    # save to csv file for plotting
    np.savetxt("data/output_data/"+scurve_filename+"_fast.csv", senscurve, delimiter=",")

    # repeat for EPTA parameters assuming a standard error of 500 ns
    psr_listEPTA = ['J0030+0451','J0613-0200','J0751+1807','J0900-3144','J1012+5307','J1022+1001','J1024-0719','J1455-3330','J1600-3053','J1640+2224','J1713+0747','J1730-2304','J1738+0333','J1744-1134','J1751-2857','J1801-1417','J1804-2717','J1843-1113','J1857+0943','J1909-3744','J1910+1256','J1911+1347','J1918-0642','J2124-3358','J2322+2057']
    freqs, scurve1  = gwb_PLSC(psr_list=psr_listEPTA, Tobs=24, cadence=(20/365.), snr=1, rmserr=1000e-9)
    freqs, scurve5  = gwb_PLSC(psr_list=psr_listEPTA, Tobs=24, cadence=(20/365.), snr=5, rmserr=1000e-9)
    freqs, scurve10 = gwb_PLSC(psr_list=psr_listEPTA, Tobs=24, cadence=(20/365.), snr=10, rmserr=1000e-9)
    senscurve = np.asarray([freqs, scurve1, scurve5, scurve10])

    # save to csv file for plotting
    np.savetxt("data/output_data/epta_scurve.csv", senscurve, delimiter=",")

    # compute skymap sensitivity for WSRT
    psr_coords, sky_coords, skymap_freqs, skymap_hc = sky_sens(psr_list=psr_list, Tobs=Tobs, cadence=cadence, rmserr=rmserr_wsrt)

    # save WSRT skymap array to csv
    np.savetxt("data/output_data/"+skymap_filename+"_wsrt.csv", skymap_hc, delimiter=",")

    # save psr coordinates
    np.savetxt("data/output_data/psr_coords.csv", psr_coords, delimiter=",")

    # save sky coordinates
    np.savetxt("data/output_data/sky_coords.csv", sky_coords, delimiter=",")

    # save frequency range
    np.savetxt("data/output_data/freqrange.csv", skymap_freqs, delimiter=",")

    # compute skymap sensitivity for FAST
    psr_coords, sky_coords, skymap_freqs, skymap_hc = sky_sens(psr_list=psr_list, Tobs=Tobs, cadence=cadence, rmserr=rmserr_fast)

    # possesses the same coords and freq range as WSRT, so just save FAST skymap array to csv
    np.savetxt("data/output_data/"+skymap_filename+"_fast.csv", skymap_hc, delimiter=",")



def gwb_PLSC(psr_list: list, Tobs: int, cadence: int, snr: int, rmserr):
    '''
    Computes power-law sensitivity curve for the GWB
    
    psr_list :  list of pulsars in timing array
    Tobs :      Total observation time of PTA campaign
    cadence :   cadence of ToAs for each pulsar (we assume all pulsars have same ToA cadence)
    snr :       desired SNR of a GW Background to compute sensitivity to
    rmserr :    the radiometer error (we assume radiometer error only)
    '''

    npsrs = len(psr_list)

    catalog = psrcat(".")

    phi = np.empty(npsrs)
    theta = np.empty(npsrs)

    for i in range(npsrs):

        psr = psr_list[i]
        ephem = catalog.get(psr)

        # get coords of pulsar from ATNF
        coords = np.asarray([ephem["ELONG"],ephem["ELAT"]], dtype=float) * (np.pi / 180)
        phi[i], theta[i] = coords

    nbins = 500 # number of frequency bins
    timespan  = [Tobs for ii in range(npsrs)] # time observing each pulsar (in this case all Tobs)
    freqs = np.logspace(np.log10(1/(Tobs*YEAR)),np.log10((cadence*365*Tobs)/(2*Tobs*YEAR)),nbins) # frequency range [1/Tobs, nToAs/2Tobs]

    # create PTA sim adding desired noise
    pta  = hasim.sim_pta(timespan=timespan,  cad=cadence*365, sigma=rmserr, phi=phi, theta=theta, freqs=freqs)
 
    spectra = []
    for psr in pta:
        sptrm = hasen.Spectrum(psr, freqs=freqs)
        sptrm.NcalInv # computes inverse-noise-weighted transmission function for each pulsar
        spectra.append(sptrm)

    # compute sensitivty curve
    senscurve = hasen.GWBSensitivityCurve(spectra)
    # compute the strain amplitude power law curve, N is number of spectral indices sampled
    pl_curve, powerlaw = hasen.PI_hc(freqs=senscurve.freqs, Tspan=senscurve.Tspan, SNR=snr, S_eff=senscurve.S_eff, N=30)
    return senscurve.freqs, pl_curve



def rms_error(tel_terms: dict, pulsar_terms: dict, t_int:float, T_sky = 125, beta = 1.25):
    """
    Compute rms error from observatory (in seconds) using the relation toa_err ~ pulse_width/SNR
    See source code for list of required terms
    
    tel_terms :      A dictionary object containing the relavent telescope properties
    pulsar_terms :   A dictionary containing the relavent pulsar properties
    t_int :          A float of the amount of observation time to create a single ToA
    T_sky :          Sky temperature in K
    X :              Some factor accounting for your choice in pulse width (default is 1.25 as per Evan Keane's guesstimate)
    """
    Kb = 1.3806503e-23 # boltzmann constant

    # telescope terms
    T_ant   = tel_terms['T_ant']    # system temp (K)
    Aeff    = tel_terms['Aeff']      # effective area of telescope (m^2) 
    n_pol   = tel_terms['npol']      # number of polarisations (x and y)
    bw      = tel_terms['bw']          # observing bandwidth (Hz)

    # pulsar terms
    w_50    = float(pulsar_terms['W50']) * 1e-3                # pulsar width at 50 percent of maximum (given in ms but converted to s)
    p       = float(pulsar_terms['P0'])                            # period (s)
    S_mean  = float(pulsar_terms['S400']) * 1e-3 * 1e-26     # mean intensity at obs frequency (400MHz) (given in mJy, converted first to Jy and then to SI units)

    duty    = w_50/p                       # duty cycle
    T_sys   = T_sky + T_ant                   # total temp
    G       = Aeff / (2 * Kb)                 # compute the gain

    # derived from radiometer equation and sigma_toa ~ width / SNR
    return beta * (T_sys / (G * S_mean * np.sqrt(t_int * n_pol * bw))) * np.sqrt(w_50**3 /(p - w_50))



def sky_sens(psr_list: list, Tobs: int, cadence: int, rmserr):
    '''
    computes sky map of GW sensitivity curve for a PTA
    
    psr_list :  list of pulsars in timing array
    Tobs :      Total observation time of PTA campaign
    cadence :   cadence of ToAs for each pulsar (we assume all pulsars have same ToA cadence)
    rmserr :    the radiometer error (we assume radiometer error only)
    '''
    
    npsrs = len(psr_list)

    catalog = psrcat(".")

    phi = np.empty(npsrs)
    theta = np.empty(npsrs)
    err = np.empty(npsrs)

    for i in range(npsrs):

        psr = psr_list[i]
        ephem = catalog.get(psr)

        # get coords of pulsar from ATNF
        coords = np.asarray([ephem["ELONG"],ephem["ELAT"]], dtype=float) * (np.pi / 180)
        phi[i], theta[i] = coords

    nbins = 81 # number of frequency bins
    timespan  = [Tobs for ii in range(npsrs)] # time observing each pulsar (in this case all Tobs)
    freqs = np.logspace(-10,-2,nbins) # frequency range

    # create PTA sim adding desired noise
    pta  = hasim.sim_pta(timespan=timespan,  cad=cadence*365, sigma=rmserr, phi=phi, theta=theta, freqs=freqs)
    
    spectra = []
    for psr in pta:
        spectrum = hasen.Spectrum(psr, freqs=freqs)
        spectrum.NcalInv # computes inverse-noise-weighted transmission function for each pulsar
        spectra.append(spectrum)

    # resolution of the sky map
    NSIDE = 32
    NPIX = hp.nside2npix(NSIDE)
    IPIX = np.arange(NPIX)
    theta_gw, phi_gw = hp.pix2ang(nside=NSIDE,ipix=IPIX)

    # create skymap
    skymap = hasky.SkySensitivity(spectra, theta_gw, phi_gw)

    # swap axes to get desired format
    skymap_Seff = np.swapaxes( np.asarray([(skymap.S_effSky[:,i]) for i in range(NPIX)]) ,0,1)

    # return the locations of each pulsar, the array of sy coordinates, the frequency array and the power spectral density skymap
    return np.asarray([theta, phi]), np.asarray([theta_gw, phi_gw]), freqs, skymap_Seff

def ska_dsa_curve():
    

if __name__ == "__main__":
    main()