# script to pull pulsar ephemerides from the ATNF pulsar catalog
# Conor Sheridan 21/04/2026

import requests
import json
from bs4 import BeautifulSoup
import os
import sys

class psrcat:
    def __init__(self, cache, version="2.7.0"):
        self.version = version

        if os.path.exists(cache) == False:
            print(f"Warniing cache does not exist at {cache}")
            sys.exit()
        else:
            if cache[-1] != "/":
                cache += "/"
            self.cache = cache + f"psrcat{version}/"

            # ensure version specific cache exists, otherwise create it
            if os.path.exists(self.cache) == False:
                os.mkdir(self.cache)

    # get ephemeris either from ATNF or cache
    def get(self, psr, force_rewrite = False):
        # check cache
        if os.path.exists(self.cache + psr + "_ephem.json") and force_rewrite == False:
            with open(self.cache + psr + "_ephem.json", 'r') as fp:
                ephem = json.load(fp)
            return ephem
        
        # otherwise pull from the ANTF
        else:
            if "+" in psr:
                psr = psr.split("+")
                url = f"https://www.atnf.csiro.au/research/pulsar/psrcat/proc_form.php?version={self.version}&startUserDefined=true&sort_attr=jname&sort_order=asc&condition=&coords_unit=raj%2Fdecj&radius=&coords_1=&coords_2=&pulsar_names={psr[0]}%2B{psr[1]}&ephemeris=long&ephemeris_submit=&style=long+with+errors&no_value=*&fsize=3&x_axis=&x_scale=linear&y_axis=&y_scale=linear&state=query"

            else:
                if "\u2212" in psr:
                    psr = psr.split("\u2212")
                elif "-" in psr:
                    psr = psr.split("-")
                url = f"https://www.atnf.csiro.au/research/pulsar/psrcat/proc_form.php?version={self.version}&startUserDefined=true&sort_attr=jname&sort_order=asc&condition=&coords_unit=raj%2Fdecj&radius=&coords_1=&coords_2=&pulsar_names={psr[0]}%2D{psr[1]}&ephemeris=long&ephemeris_submit=&style=long+with+errors&no_value=*&fsize=3&x_axis=&x_scale=linear&y_axis=&y_scale=linear&state=query"

            res = requests.get(url)
            soup = BeautifulSoup(res.content, 'html.parser')

            content = soup.find('pre').string.split("\n")
            
            return {i.split()[0]:i.split()[1] for i in content[1:-1]}
    
    def update_cache():
        # to be written
        pass

    def write(self, ephem, dir=None):
        output = ephem["PSRJ"] + "_ephem.json"

        # write output to cache
        with open(self.cache+output, 'w+') as fp:
            json.dump(ephem, fp)

        # if desired copy output to given directory
        if dir != None:
            with open(dir+output, 'w+') as fp:
                json.dump(ephem, fp)


    def read(self, dir, psr):
        ephemfile = dir + psr + "_ephem.json"

        if os.path.exists(ephemfile):
            with open(ephemfile, 'r') as fp:
                ephem = json.load(fp)
            return ephem
        
        else:
            raise(f"Ephemeris at {dir}{psr} does not exist, use psrcat.get() instead!")

def main():
    npsr = len(sys.argv) - 2
    if len(sys.argv) <= 0:
        print("Usage: python script.py <cache> <psr1> <psr2> ... <psrn>")
        sys.exit(1)

    cache = sys.argv[1]
    psrcat0 = psrcat(cache) # initialise pulsar catalog

    for i in range(npsr):
        psr = sys.argv[i+2]
        print(f"Getting ephemeris for {psr}...")
        ephem = psrcat0.get(psr)
        psrcat0.write(ephem,".")
    print("done\n")

if __name__ == "__main__":
    main()