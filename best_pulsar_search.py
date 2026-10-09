from sensitivity_curve import rms_error
import json
import sys
from atnf_pull import psrcat

def sortpsrs(psr_list: list, toa_int: int, tel_params: str):
    '''
    sort and rank pulsars based on how small their radiometer error is.

    psr_list    : list of pulsars to compute errors for and sort
    toa_int     : ToA integration time for each pulsar
    tel_params  : path string to a .json file containing the required telescope parameters to compute the error for
    '''

    # define local ANTF catalog directory 
    catalog = psrcat(".")

    # observatory parameters
    with open(tel_params, 'r') as fp:
        wsrt_params = json.load(fp)

    # compute rmserr for each psr and add to list
    pulsar_rms = {}
    for psr in psr_list:
        ephem = catalog.get(psr) # get ephemeris for a given pulsar
        try: # try compute radiometer error, if psrcat doesn't have all the required paramters it will give an error
            rmserr_wsrt = rms_error(wsrt_params, ephem, toa_int) 
            pulsar_rms.update({psr:rmserr_wsrt})
        except KeyError: # if a key error is raised, one or more of the required parameters are not available
            pass
        except:
            rmserr_wsrt = rms_error(wsrt_params, ephem, toa_int)

    # sort the dictionary by rms_err size so smallest first and return
    return dict(sorted(pulsar_rms.items(), key=lambda item: item[1]))

def save_rms_dict(rms_err: dict, dir="."):
    '''
    save the error values dictionary as a .json file

    rms_err : dictionary of radiometer error to save
    dir     : directory to save it to 
    '''
        
    # if desired copy output to given directory
    if dir != None:
        with open(dir+"rms_err", 'w+') as fp:
            json.dump(rms_err, fp)

def main():
    '''python3 best_pulsar_search.py <desired_npsr>
    
    desired_npsr : number of pulsars to print, if left blank will print all pulsars (that it has values for)
    '''

    args = sys.argv[1:]

    if len(args) == 0: 
        desired_npsr = True # if left blank, mark as true for now
    elif "-h" in args or "--help" in args:
        print("Python script to compute the radiometer error of a the EPTA DR2 pulsars and ranks them.\n\nRun as : python3 best_pulsar_Search.py <desired_npsrs>\n\ndesired_npsrs :\tnumber of pulsars to display, if no value is entered display all\n")
        exit(0)
    elif len(args) > 1:
        raise TypeError("Too many arguments given!\n\nRun as : python3 best_pulsar_Search.py <desired_npsrs>\n")
    else:
        desired_npsr = int(args[0])
        if desired_npsr<1:
            raise TypeError("Must be a positive number")

    # every Pulsar in the EPTA DR2
    psr_list = ['J0030+0451','J0613-0200','J0751+1807','J0900-3144','J1012+5307','J1022+1001','J1024-0719','J1455-3330','J1600-3053','J1640+2224','J1713+0747','J1730-2304','J1738+0333','J1744-1134','J1751-2857','J1801-1417','J1804-2717','J1843-1113','J1857+0943','J1909-3744','J1910+1256','J1911+1347','J1918-0642','J2124-3358','J2322+2057']
    wsrt_params = "./data/observatories/wsrt_params.json"


    # get ToA integration time from pta_config file
    with open("./pta_config.json", 'r') as fp:
        pta_config = json.load(fp)
    toa_int = pta_config["toa_int"]
    
    psrerr = sortpsrs(psr_list, toa_int, wsrt_params) # compute each pulsars radiometer err

    # if marked as true then all pulsars should be printed
    if desired_npsr == True:
        desired_npsr = len(psrerr)

    # print these best pulsars
    print(f"Printing top {desired_npsr} pulsars by rms error\n\nPULSAR\t\tRMS ERROR")
    for k in range(desired_npsr):
        psr = list(psrerr)[k]
        print(f"{psr}\t{psrerr[psr]}")

if __name__ == '__main__':
    main()