#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import toml

from .opDB import OpDB
from .qaDB import QaDB
from .utils import read_conf

TEXP_NOMINAL = 900.0         # sec.
SEEING_NOMINAL = 0.80        # arcsec
TRANSPARENCY_NOMINAL = 0.90  # 
NOISE_LEVEL_NOMINAL = 10.0   # ADU?
THROUGHPUT_NOMINAL = 0.15    #

__all__ = ["ExposureTime"]

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
        self.THROUGHPUT_NOMINAL = THROUGHPUT_NOMINAL
        self.object_faintness = object_faintness
        self.df_fae = self.getFiberApertureEffectModel()

        # database config
        self.opdb = OpDB(self.conf["db"]["opdb"])
        self.qadb = QaDB(self.conf["db"]["qadb"])

    def getNominalExposureTime(self, visit):
        """        
        Parameters
        ----------
            visit : `int` pfs_visit_id

        Returns
        ----------
            t_nominal: `float` (nominal exposure time in sec.)

        Examples
        ----------

        """
        sqlWhere = f"sps_exposure.pfs_visit_id={visit}"
        sqlCmd = f"SELECT * FROM sps_exposure WHERE {sqlWhere};"
        df = pd.read_sql(sql=sqlCmd, con=self.opdb._conn)
        self.t_nominal = np.nanmean(df["exptime"])
        return self.t_nominal
    
    def getFiberApertureEffectModel(self):
        """ Read csv file of the fiber aperture effect
        The file contains the fraction of light flux covered by the fiber aperture as a function of seeing FWHM in arcsec.

        Parameters
        ----------

        Returns
        ----------
            df: DataFrame of fiber aperture effect

        Examples
        ----------

        """
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
    
    def calcEffectiveExposureTime(self, visit, seeing, transparency, throughput, noise_level, mode='FLUXSTD'):
        """        
        Parameters
        ----------
            visit : `int` pfs_visit_id
            seeing : `float` seeing FWHM in arcsec.
            transparency : `float` transparency
            throughput : `float` throughput
            noise_level : `float` sky-background noise_level

        Returns
        ----------
            t_eff: `float` effective exposure time 

        Examples
        ----------

        """
        # calculate the relative change of the observing conditions
        if mode == 'FLUXSTD':
            xi = 1.0
            eta = throughput / self.THROUGHPUT_NOMINAL
        else:
            xi = self.calcFiberApertureEffect(seeing) / self.calcFiberApertureEffect(self.SEEING_NOMINAL)
            eta = transparency / self.TRANSPARENCY_NOMINAL
        noise = noise_level / self.NOISE_LEVEL_NOMINAL
        
        # calculate the effective exposure time
        if self.object_faintness == 0:
            # object is bright (so object limited)
            self.t_effective = self.TEXP_NOMINAL * xi * eta
        else:
            # object is faint (so background limited)
            self.t_effective = self.TEXP_NOMINAL / noise**2 * xi**2 * eta**2

        # insert into qaDB
        df = pd.DataFrame(
            data={"pfs_visit_id": [visit],
                  "nominal_exposure_time": [self.t_nominal],
                  "effective_exposure_time": [self.t_effective]
                  })
        self.qadb.populateQATable('exposure_time', df)

        return self.t_effective

