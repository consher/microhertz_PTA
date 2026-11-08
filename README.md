# Predicting the Sensitivity of a PTA targetting Microhertz GW Frequnecies
This is the accompanying code for this paper [link]. We used the [`hasasia`](https://hasasia.readthedocs.io/) python library to predict the sensitivty of a Pulsar Timing Array with a sub-daily candence of ToA measurements, attempting to target the microhertz GW frequency band.

* `sensitivity_curve.py` computes the power-law and sky-map sensitivity curves (radiometer noise limited) of a sub-daily cadence PTA configurable in `pta_config.json` assuming the use of Westerbork Synthesis Radio Telescope & the Five-hundred metre Aperture Spherical Telescope. IT also creates a becnhmark EPTA DR2 power-law sensitivity curve.
* `best_pulsar_search.py` computes the radiometer error for the EPTA DR2 pulsars given `pta_config.json` and WSRT parameters (`data/observatories/wsrt_params.json`), then ranks them.
* `plotting.py` produces plots used in paper.
* `atnf_pull.py` homemade script to pull and read pulsar ephemerides from the [ATNF Pulsar Catalog](https://www.atnf.csiro.au/research/pulsar/psrcat/) and stores them in `./psrcat2.7.0`.

The directory `./data` contains:
* `/new-physics_data`: contains new-physics data taken from [Blanco-Pillado & Olum (2017)](http://doi.org/10.1103/PhysRevD.96.104046) and [Renzini et al. (2022)](http://doi.org/10.3390/galaxies10010034). We get domain wall values from simulation papers (e.g. [Kawasaki & Saikawa 2011](http://doi.org/10.1088/1475-7516/2011/09/008)).
* `/observatories`: observatory config `.json` files.

Output data is written to a folder not in this repo, the data generated for this paper can be accessed at [10.5281/zenodo.23690555](https://doi.org/10.5281/zenodo.23690555).