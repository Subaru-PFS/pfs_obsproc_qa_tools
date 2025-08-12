#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import toml

from .opDB import OpDB
from .qaDB import QaDB
from .utils import read_conf

__all__ = ["ExposureTime"]

class ExposureTime(object):
    """Exposure Time

    The exposure time required to achieve a given S/N (SNR) is defined as follows

            t = SNR^2 x (N_obj x \\xi + N_sky) / (N_obj^2 x \\xi^2 x \\eta),

            where N_obj is the count rate of object and N_sky is the count rate of sky within the fiber aperture. \\xi is the flux ratio degraded by the fiber aperture and \\eta is the throughput including instrument and atmosphere. Note that \\xi depends on seeing and \\eta depends on the atmospheric transparency.

        (1) bright objects compared to sky background

            t = SNR^2 x 1 / (N_obj x \\xi x \\eta)

        (2) faint objects compared to sky background

            t = SNR^2 x N_sky / (N_obj^2 x \\xi^2 x \\eta^2)
    
    The effective exposure time, therefore, is defined as follows

        (1) bright objects compared to sky background

            t_eff = t_nominal x (\\xi_obs/\\xi_nom) x (\\eta_obs/\\eta_nom)

            (if \\xi is included in the throughput term)
            t_eff = t_nominal x (\\eta_obs/\\eta_nom)

        (2) faint objects compared to sky background

            t_eff = t_nominal x (N_sky,nom/N_sky,obs) x (\\xi_obs/\\xi_nom)^2 x (\\eta_obs/\\eta_nom)^2
                or
            t_eff = t_nominal x (sigma_sky,nom/sigma_sky,obs)^2 x (\\xi_obs/\\xi_nom)^2 x (\\eta_obs/\\eta_nom)^2

            (if \\xi is included in the throughput term)
            t_eff = t_nominal x (sigma_sky,nom/sigma_sky,obs)^2 x (\\eta_obs/\\eta_nom)^2
            
    """
    def __init__(self, conf='config.toml', isFaint=True):
        """        
        Parameters
        ----------
            conf: `str` config toml filepath (default: config.toml)
            isFaint : `bool` (default: True)
        
        Returns
        ----------
            config: return of toml.load
        
        Examples
        ----------

        """

        # load configuration
        self.conf = read_conf(conf)

        # nominal values etc.
        self.TEXP_REF = self.conf['qa']['exptime']['nominal']
        self.TEXP_NOMINAL = self.conf['qa']['exptime']['nominal']
        self.TEXP_MAX = self.conf['qa']['exptime']['max_effective']        
        self.TEXP_MAX_SCALE = self.conf['qa']['exptime']['max_effective_scale']        
        self.SEEING_NOMINAL = self.conf['qa']['seeing']['nominal']
        self.TRANSPARENCY_NOMINAL = self.conf['qa']['transparency']['nominal']
        self.NOISE_LEVEL_NOMINAL_B = self.conf['qa']['sky']['noise_nominal_b']
        self.NOISE_LEVEL_NOMINAL_R = self.conf['qa']['sky']['noise_nominal_r']
        self.NOISE_LEVEL_NOMINAL_N = self.conf['qa']['sky']['noise_nominal_n']
        self.NOISE_LEVEL_NOMINAL_M = self.conf['qa']['sky']['noise_nominal_m']
        self.BACKGROUND_LEVEL_NOMINAL = self.conf['qa']['sky']['background_nominal']
        self.THROUGHPUT_NOMINAL_B = self.conf['qa']['throughput']['throughput_nominal_b']
        self.THROUGHPUT_NOMINAL_R = self.conf['qa']['throughput']['throughput_nominal_r']
        self.THROUGHPUT_NOMINAL_N = self.conf['qa']['throughput']['throughput_nominal_n']
        self.THROUGHPUT_NOMINAL_M = self.conf['qa']['throughput']['throughput_nominal_m']
        self.isFaint = isFaint
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
            TEXP_NOMINAL: `float` (nominal exposure time in sec.)

        Examples
        ----------

        """
        sqlWhere = f"sps_exposure.pfs_visit_id={visit}"
        sqlCmd = f"SELECT * FROM sps_exposure WHERE {sqlWhere};"
        df = pd.read_sql(sql=sqlCmd, con=self.opdb._conn)
        self.TEXP_NOMINAL = np.nanmean(df["exptime"])
        return self.TEXP_NOMINAL
    
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
    
    def calcEffectiveExposureTime(self, visit, seeing, transparency, 
                                  throughput_b, throughput_r, throughput_n, throughput_m, 
                                  background_level, noise_level_b, noise_level_r, noise_level_n, noise_level_m, 
                                  useBackground=False, useFluxstd=True, updateDB=True):
        """
        Parameters
        ----------
            visit : `int` pfs_visit_id
            seeing : `float` seeing FWHM in arcsec.
            transparency : `float` transparency
            throughput_b : `float` throughput in b-arm
            throughput_r : `float` throughput in r-arm
            throughput_n : `float` throughput in n-arm
            throughput_m : `float` throughput in m-arm
            background_level : `float` sky background level in r-arm
            noise_level_b : `float` sky noise_level in b-arm
            noise_level_r : `float` sky noise_level in r-arm
            noise_level_n : `float` sky noise_level in n-arm
            noise_level_m : `float` sky noise_level in m-arm
            useBackground : use background_level to calculate EET? (default=False)
            useFluxstd : use throughput from FLUXSTDs to calculate EET? (default=True)
            updateDB : whether or not update DB (default=True)
        Returns
        ----------
            t_eff: `float` effective exposure time 

        Examples
        ----------

        """
        # calculate the relative change of the observing conditions
        if useFluxstd is True:
            xi = 1.0
            eta_b = throughput_b / self.THROUGHPUT_NOMINAL_B
            eta_r = throughput_r / self.THROUGHPUT_NOMINAL_R
            eta_n = throughput_n / self.THROUGHPUT_NOMINAL_N
            eta_m = throughput_m / self.THROUGHPUT_NOMINAL_M
        else:
            xi = self.calcFiberApertureEffect(seeing) / self.calcFiberApertureEffect(self.SEEING_NOMINAL)
            eta_b = transparency / self.TRANSPARENCY_NOMINAL
            eta_r = transparency / self.TRANSPARENCY_NOMINAL
            eta_n = transparency / self.TRANSPARENCY_NOMINAL
            eta_m = transparency / self.TRANSPARENCY_NOMINAL
        noise_b = noise_level_b / self.NOISE_LEVEL_NOMINAL_B
        noise_r = noise_level_r / self.NOISE_LEVEL_NOMINAL_R
        noise_n = noise_level_n / self.NOISE_LEVEL_NOMINAL_N
        noise_m = noise_level_m / self.NOISE_LEVEL_NOMINAL_M
        background = background_level / self.BACKGROUND_LEVEL_NOMINAL
        
        # calculate the effective exposure time
        if self.isFaint is False:
            # object is bright (so object limited)
            self.t_effective_b = self.TEXP_NOMINAL * xi * eta_b
            self.t_effective_r = self.TEXP_NOMINAL * xi * eta_r
            self.t_effective_n = self.TEXP_NOMINAL * xi * eta_n
            self.t_effective_m = self.TEXP_NOMINAL * xi * eta_m
        else:
            # object is faint (so background limited)
            if useBackground is True:
                self.t_effective_b = self.TEXP_NOMINAL / background * xi**2 * eta_b**2
                self.t_effective_r = self.TEXP_NOMINAL / background * xi**2 * eta_r**2
                self.t_effective_n = self.TEXP_NOMINAL / background * xi**2 * eta_n**2
                self.t_effective_m = self.TEXP_NOMINAL / background * xi**2 * eta_m**2
            else:
                self.t_effective_b = self.TEXP_NOMINAL / noise_b**2 * xi**2 * eta_b**2
                self.t_effective_r = self.TEXP_NOMINAL / noise_r**2 * xi**2 * eta_r**2
                self.t_effective_n = self.TEXP_NOMINAL / noise_n**2 * xi**2 * eta_n**2
                self.t_effective_m = self.TEXP_NOMINAL / noise_m**2 * xi**2 * eta_m**2

        # apply a cap for effective exposure time
        if self.t_effective_b > self.TEXP_NOMINAL * self.TEXP_MAX_SCALE:
            self.t_effective_b = self.TEXP_NOMINAL * self.TEXP_MAX_SCALE
        if self.t_effective_r > self.TEXP_NOMINAL * self.TEXP_MAX_SCALE:
            self.t_effective_r = self.TEXP_NOMINAL * self.TEXP_MAX_SCALE
        if self.t_effective_n > self.TEXP_NOMINAL * self.TEXP_MAX_SCALE:
            self.t_effective_n = self.TEXP_NOMINAL * self.TEXP_MAX_SCALE
        if self.t_effective_m > self.TEXP_NOMINAL * self.TEXP_MAX_SCALE:
            self.t_effective_m = self.TEXP_NOMINAL * self.TEXP_MAX_SCALE

        # insert into qaDB
        df = pd.DataFrame(
            data={"pfs_visit_id": [visit],
                  "nominal_exposure_time": [self.TEXP_NOMINAL],
                  "effective_exposure_time_b": [self.t_effective_b],
                  "effective_exposure_time_r": [self.t_effective_r],
                  "effective_exposure_time_n": [self.t_effective_n],
                  "effective_exposure_time_m": [self.t_effective_m],
                  })
        self.qadb.populateQATable('exposure_time', df, updateDB=updateDB)

        return (self.t_effective_b, self.t_effective_r, self.t_effective_n, self.t_effective_m)

