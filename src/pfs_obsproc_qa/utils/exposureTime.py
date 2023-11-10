#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import toml

from .opDB import OpDB
from .qaDB import QaDB

TEXP_NOMINAL = 900.0         # sec.
SEEING_NOMINAL = 0.80        # arcsec
TRANSPARENCY_NOMINAL = 0.90  # 
NOISE_LEVEL_NOMINAL = 10.0   # ADU?

__all__ = ["ExposureTime"]


def read_conf(conf):
    """
    Parameters
    ----------
        conf: `str` config toml filepath (default: config.toml)
    
    Returns
    ----------
        config: return of toml.load
    
    Examples
    ----------

    """
    config = toml.load(conf)
    return config

class ExposureTime(object):
    """Exposure Time

    The exposure time required to achieve a given S/N (SNR) is defined as follows

            t = SNR^2 x (N_obj x \\xi + N_sky) / (N_obj^2 x \\xi^2 x \\eta)

        (1) bright objects compared to sky background

            t = SNR^2 x 1 / (N_obj x \\xi x \\eta)

        (2) faint objects compared to sky background

            t = SNR^2 x N_sky / (N_obj^2 x \\xi^2 x \\eta^2)
    
    The effective exposure time, therefore, is defined as follows

        (1) bright objects compared to sky background

            t_eff = t_nominal x (\\xi_obs/\\xi_nom) x (\\eta_obs/\\eta_nom)

        (2) faint objects compared to sky background

            t_eff = t_nominal x (N_sky,nom/N_sky,obs) x (\\xi_obs/\\xi_nom)^2 x (\\eta_obs/\\eta_nom)^2
                or
            t_eff = t_nominal x (sigma_sky,nom/sigma_sky,obs)^2 x (\\xi_obs/\\xi_nom)^2 x (\\eta_obs/\\eta_nom)^2

    """
    def __init__(self, conf='config.toml', object_faintness=1):
        """        
        Parameters
        ----------
            conf: `str` config toml filepath (default: config.toml)
            object_faintness : `int` (0=bright, 1=faint, default=1)
        
        Returns
        ----------
            config: return of toml.load
        
        Examples
        ----------

        """

        # load configuration
        self.conf = read_conf(conf)       

        # nominal values etc.
        self.TEXP_NOMINAL = TEXP_NOMINAL
        self.SEEING_NOMINAL = SEEING_NOMINAL
        self.TRANSPARENCY_NOMINAL = TRANSPARENCY_NOMINAL
        self.NOISE_LEVEL_NOMINAL = NOISE_LEVEL_NOMINAL
        self.object_faintness = object_faintness
        self.df_fae = self.getFiberApertureEffectModel()

        # database config
        self.opdb = OpDB(self.conf["db"]["opdb"])
        self.qadb = QaDB(self.conf["db"]["qadb"])

    def getFiberApertureEffectModel(self):
        df = pd.read_csv(os.path.join(__file__.split('utils')[0], 'data/fiber_aperture_effect.csv'))
        return df
    
    def calcFiberApertureEffect(self, seeing):
        """        
        Parameters
        ----------
            seeing : `float` seeing FWHM in arcsec.

        Returns
        ----------
            fae: fiber aperture effect (fraction of flux covered by the fiber aperture)

        Examples
        ----------

        """
        fae = np.interp(seeing, self.df_fae['fwhm'], self.df_fae['ap_eff_moffat'])
        return fae
    
    def calcEffectiveExposureTime(self, seeing, transparency, noise_level):
        """        
        Parameters
        ----------
            seeing : `float` seeing FWHM in arcsec.
            transparency : `float` transparency
            noise_level : `float` sky-background noise_level

        Returns
        ----------
            t_eff: `float` effective exposure time 

        Examples
        ----------

        """
        xi = self.calcFiberApertureEffect(seeing) / self.calcFiberApertureEffect(self.SEEING_NOMINAL)
        eta = transparency / self.TRANSPARENCY_NOMINAL
        noise = noise_level / self.NOISE_LEVEL_NOMINAL
        
        if self.object_faintness == 0:
            # object is bright (so object limited)
            t_eff = self.TEXP_NOMINAL * xi * eta
        else:
            # object is faint (so background limited)
            t_eff = self.TEXP_NOMINAL / noise**2 * xi**2 * eta**2

        return t_eff

