#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import toml

TEXP_NOMINAL = 900.0         # sec.
SEEING_NOMINAL = 0.80        # arcsec
TRANSPARENCY_NOMINAL = 0.90  # 
NOISE_LEVEL_NOMINAL = 40.0   # ADU?

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

            t_eff = t_nominal / (N_sky,obs/N_sky,nom) x (\\xi_obs/\\xi_nom)^2 x (\\eta_obs/\\eta_nom)^2

    Parameters
    ----------
    object_faintness : `int` (0=bright, 1=faint, default=1)

    Examples
    ----------

    """
    def __init__(self, object_faintness=1):
        self.TEXP_NOMINAL = TEXP_NOMINAL
        self.SEEING_NOMINAL = SEEING_NOMINAL
        self.TRANSPARENCY_NOMINAL = TRANSPARENCY_NOMINAL
        self.NOISE_LEVEL_NOMINAL = NOISE_LEVEL_NOMINAL
        self.object_faintness = object_faintness
        self.df_fae = self.getFiberApertureEffectModel()

    def getFiberApertureEffectModel(self):
        df = pd.read_csv(os.path.join(__file__.split('utils')[0], 'data/fiber_aperture_effect.csv'))
        return df
    
    def calcFiberApertureEffect(self, seeing):
        fae = np.interp(seeing, self.df_fae['fwhm'], self.df_fae['ap_eff_moffat'])
        return fae
    
    def calcEffectiveExposureTime(self, seeing, transparency, noise_level):
        xi = self.calcFiberApertureEffect(seeing) / self.calcFiberApertureEffect(self.SEEING_NOMINAL)
        eta = transparency / self.TRANSPARENCY_NOMINAL
        noise = noise_level / self.NOISE_LEVEL_NOMINAL
        
        if self.object_faintness == 0:
            # object is bright (so object limited)
            t_eff = self.TEXP_NOMINAL * xi * eta
        else:
            # object is faint (so background limited)
            t_eff = self.TEXP_NOMINAL / noise * xi**2 * eta**2

        return t_eff

