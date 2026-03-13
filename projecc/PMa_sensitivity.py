#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 17:56:29 2021

@author: kervella

Updated on 10/29/2023 by Logan Pearce

For a single object, query the Kervella 2022 catalog of proper motion
anomaly and generate a plot of corresponding companion masses as a function
of separation as seen in Kervella et al. 2022

Usage: python PMa_sensitivity.py 'NAME'
where NAME is a Simbad-resolvable name of object.

"""
import argparse
from importlib.resources import files

import matplotlib.pyplot as plt
import numpy as np
from astroquery.vizier import Vizier
from astropy import constants as const
from astropy import units as u
from astropy.io import ascii


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Query Kervella 2022 PMa table for a target and generate "
            "a companion-mass sensitivity curve."
        )
    )
    parser.add_argument(
        "SimbadNAME",
        metavar="N",
        help="Simbad-resolvable object name",
    )
    return parser.parse_args(argv)

# Time window smearing efficiency function
def gamfunc(P):
    gam = P/(np.sqrt(2)* np.pi) * np.sqrt(1 - np.cos(2*np.pi/P))
    return gam

# Hip-GDR2 time base smearing efficiency function interpolation
def zetafunc(P,zetval):
    zet = np.interp(np.array(P), np.array(zetval['P/T']), np.array(zetval['zeta']))
    zet_errplus = np.interp(np.array(P), np.array(zetval['P/T']), np.array(zetval['zeta_errplus']))
    zet_errminus = np.interp(np.array(P), np.array(zetval['P/T']), np.array(zetval['zeta_errminus']))
    return np.array(zet), np.array(zet_errplus), np.array(zet_errminus)

# PMa sensitivity function secondary mass-orbital radius diagram
def mBr_function(starmass, dvel_norm, dvel_norm_err,\
                 dtGaia=(1037.93*u.day),dtHG=((2016.0-1991.25)*u.year)):
    # dtHG = 2016.0 - 1991.25
    # dtGaia = (Time('2017-05-28T08:44:00',format='isot')-Time('2014-07-25T10:30:00', format='isot'))
    #        = 1037.93 days

    zeta_path = files("projecc").joinpath("zeta-values.csv")
    zeta = ascii.read(str(zeta_path))
    minAU = 0.5
    maxAU = 200
    nval = 500
    r = np.geomspace(minAU,maxAU,nval)*u.au
    
    Pr = np.sqrt(r**3 * (4*np.pi**2)/(const.G*starmass))
    barP = (Pr.to(u.year).value)/(dtGaia.to(u.year).value)
    zetval, zetval_errplus, zetval_errminus = zetafunc(Pr.to(u.year).value/dtHG,zeta)
    mBr = (np.sqrt(r/u.au) / gamfunc(barP) / zetval\
        * dvel_norm/0.87\
        * np.sqrt(1*u.au * starmass/const.G)).to(u.Mjup)
    rel_Verr = dvel_norm_err / dvel_norm

    mBrmin = mBr * (1-np.sqrt((.12/.87)**2 + rel_Verr**2 + (zetval_errminus/zetval)**2))
    mBrmax = mBr * (1+np.sqrt((.32/.87)**2 + rel_Verr**2 + (zetval_errplus/zetval)**2))

    return r, mBr, mBrmin, mBrmax

def main(argv=None):
    args = parse_args(argv)
    name = args.SimbadNAME

    # Sensitivity function in terms of companion mass as a function of
    # tangential velocity anomaly from Kervella et al. 2022.
    cat = "J/A+A/657/A7/tablea1"
    result = Vizier.query_object(name, catalog=cat)
    table = result[0]
    if len(table) == 0:
        print("Object not in Kervella 2022 catalog")
        return 1

    starmass = table["M1"][0] * u.Msun
    dvel_norm = table["dVt"][0] * u.m / u.s
    dvel_norm_err = table["e_dVt"][0] * u.m / u.s
    radius, mBr, mBrmin, mBrmax = mBr_function(starmass, dvel_norm, dvel_norm_err)

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    ax1.set_title(
        r"PMa sensitivity %.1f $M_\odot$ star and $dV_\mathrm{tan} = %.2f \pm %.2f$ m/s"
        % (
            starmass.to(u.Msun).value,
            dvel_norm.to(u.m / u.s).value,
            dvel_norm_err.to(u.m / u.s).value,
        )
    )
    ax1.set_xlabel(r"Orbital radius (au)")
    ax1.set_ylabel(r"$m_2\,(M_\mathrm{Jup})$")
    ax1.grid()
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.plot(radius, mBr.to(u.Mjup).value, label=r"$m_2$ EDR3 PMa", color="forestgreen")
    ax1.fill_between(
        radius.to(u.AU).value,
        mBrmin.to(u.Mjup).value,
        mBrmax.to(u.Mjup).value,
        facecolor="limegreen",
        alpha=0.2,
    )
    fig.savefig("Kervella-plot-" + name.replace(" ", "") + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
