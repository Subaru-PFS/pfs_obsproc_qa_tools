#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import toml
import matplotlib.pyplot as plt
from pfs.datamodel import Identity, PfsArm, PfsMerged
from pfs.datamodel.pfsConfig import PfsConfig, TargetType
from pfs.datamodel.pfsFluxReference import PfsFluxReference
from .opDB import OpDB
from .qaDB import QaDB
from .utils import read_conf
import logzero
from logzero import logger

__all__ = ["Condition"]

TEXP_NOMINAL = 900.0         # sec.
SEEING_NOMINAL = 0.80        # arcsec
TRANSPARENCY_NOMINAL = 0.90  # 
NOISE_LEVEL_NOMINAL = 10.0   # ADU?
THROUGHPUT_NOMINAL = 0.15    #

class Condition(object):
    """Observing condition
    """
    
    def __init__(self, conf='config.toml'):
        """
        Parameters
        ----------
            conf: `str` config toml filepath (default: config.toml)
        
        Returns
        ----------
            
        Examples
        ----------

        """

        # load configuration
        self.conf = read_conf(conf)       

        # visit list
        self.visitList = []

        # database
        self.opdb = OpDB(self.conf["db"]["opdb"])
        self.qadb = QaDB(self.conf["db"]["qadb"])

        # dataframe for condition
        self.df_guide_error = None
        self.df_seeing = None
        self.df_transparency = None
        self.df_ag_background = None
        self.df_sky_background = None
        self.df_sky_background_stats_pv = None
        self.df_sky_noise = None
        self.df_flxstd_noise = None
        self.df_science_noise = None
        self.df_sky_noise_stats_pv = None
        self.df_seeing_stats = None
        self.df_transparency_stats = None
        self.df_ag_background_stats = None
        self.df_noise_stats = None
        self.df_seeing_stats_pv = None
        self.df_transparency_stats_pv = None
        self.df_ag_background_stats_pv = None
        self.df_noise_stats_pv = None
        self.df_throughput = None
        self.df_throughput_stats_pv = None

        self.skyQaConf = self.conf["qa"]["sky"]["config"]

        logzero.logfile(self.conf['logger']['logfile'], disableStderrLogger=True)

    def getAgcData(self, visit):
        """ Get AGC data information

        Parameters
        ----------
            visit : pfs_visit_id
        
        Returns
        ----------
            
        Examples
        ----------

        """
        sqlWhere = f'agc_exposure.pfs_visit_id={visit}'
        sqlCmd = f'SELECT agc_exposure.pfs_visit_id,agc_exposure.agc_exposure_id,agc_exposure.taken_at,agc_data.agc_camera_id,central_image_moment_11_pix,central_image_moment_20_pix,central_image_moment_02_pix,background,estimated_magnitude,agc_data.flags,agc_match.guide_star_id,pfs_design_agc.guide_star_magnitude FROM agc_exposure JOIN agc_data ON agc_exposure.agc_exposure_id=agc_data.agc_exposure_id JOIN agc_match ON agc_data.agc_exposure_id=agc_match.agc_exposure_id AND agc_data.agc_camera_id=agc_match.agc_camera_id AND agc_data.spot_id=agc_match.spot_id JOIN pfs_design_agc ON agc_match.pfs_design_id=pfs_design_agc.pfs_design_id AND agc_match.guide_star_id=pfs_design_agc.guide_star_id WHERE {sqlWhere} AND agc_data.flags<=1 ORDER BY agc_exposure.agc_exposure_id;'
        df = pd.read_sql(sql=sqlCmd, con=self.opdb._conn)
        return df
    
    def getSpsExposure(self, visit):
        """ Get SpS exposure information

        Parameters
        ----------
            visit : pfs_visit_id
        
        Returns
        ----------
            
        Examples
        ----------

        """
        sqlWhere = f'sps_exposure.pfs_visit_id={visit}'
        sqlCmd = f'SELECT * FROM sps_exposure WHERE {sqlWhere};'
        df = pd.read_sql(sql=sqlCmd, con=self.opdb._conn)
        return df

    def getPfsVisit(self, visit):
        """ Get SpS exposure information

        Parameters
        ----------
            visit : pfs_visit_id
        
        Returns
        ----------
            
        Examples
        ----------

        """
        sqlWhere = f'pfs_visit.pfs_visit_id={visit}'
        sqlCmd = f'SELECT * FROM pfs_visit WHERE {sqlWhere};'
        df = pd.read_sql(sql=sqlCmd, con=self.opdb._conn)
        return df

    def getGuideError(self, visit):
        """ Get GuideError information

        Parameters
        ----------
            visit : pfs_visit_id
        
        Returns
        ----------
            
        Examples
        ----------

        """
        sqlWhere = f'agc_exposure.pfs_visit_id={visit}'
        sqlCmd = f'SELECT agc_exposure.pfs_visit_id,agc_exposure.agc_exposure_id,agc_exposure.taken_at,agc_guide_offset.guide_ra,agc_guide_offset.guide_dec,agc_guide_offset.guide_pa,agc_guide_offset.guide_delta_ra,agc_guide_offset.guide_delta_dec,agc_guide_offset.guide_delta_insrot,agc_guide_offset.guide_delta_az,agc_guide_offset.guide_delta_el,agc_guide_offset.guide_delta_z,agc_guide_offset.guide_delta_z1,agc_guide_offset.guide_delta_z2,agc_guide_offset.guide_delta_z3,agc_guide_offset.guide_delta_z4,agc_guide_offset.guide_delta_z5,agc_guide_offset.guide_delta_z6,agc_guide_offset.guide_delta_scale,agc_guide_offset.mask FROM agc_exposure JOIN agc_guide_offset ON agc_exposure.agc_exposure_id=agc_guide_offset.agc_exposure_id WHERE {sqlWhere} ORDER BY agc_exposure.agc_exposure_id;'
        df = pd.read_sql(sql=sqlCmd, con=self.opdb._conn)
        if self.df_guide_error is None:
            self.df_guide_error = df.copy()
        else:
            self.df_guide_error = pd.concat([self.df_guide_error, df], ignore_index=True)

    def calcSeeing(self, visit):
        """Calculate Seeing size based on AGC measurements

        Parameters
        ----------
            visit : pfs_visit_id
        
        Returns
        ----------
            agc_exposure_id : AGC exposure ID
            taken_at_seq    : The time at which the exposure was taken (in HST)
            fwhm_median     : The median FWHM during the AGC exposure  (arcsec.)
            
        Examples
        ----------

        """
        logger.info(f"calculating seeing for visit={visit}...")

        # get information from agc_data table
        df = self.getAgcData(visit)
        agc_exposure_id = df['agc_exposure_id']
        seq = np.unique(agc_exposure_id)
        taken_at = df['taken_at']
        pfs_visit_id = df['pfs_visit_id'].astype(int)
        _df = self.getPfsVisit(visit)
        pfs_visit_description = _df['pfs_visit_description'][0]
        pfs_design_id = _df['pfs_design_id'][0]
        issued_at = _df['issued_at'][0]
        agc_camera_id = df['agc_camera_id']
        flags = df['flags']

        # calculate FWHM (reference: Magnier et al. 2020, ApJS, 251, 5)
        g1 = df['central_image_moment_20_pix'] + df['central_image_moment_02_pix']
        g2 = df['central_image_moment_20_pix'] - df['central_image_moment_02_pix']
        g3 = np.sqrt(g2**2 + 4*df['central_image_moment_11_pix']**2)
        sigma_a = self.conf['agc']['ag_pix_scale'] * np.sqrt((g1+g3)/2)
        sigma_b = self.conf['agc']['ag_pix_scale'] * np.sqrt((g1-g3)/2)
        sigma = np.sqrt(sigma_a*sigma_b)

        # simple calculation 
        #varia = df['central_image_moment_20_pix'] * df['central_image_moment_02_pix'] - df['central_image_moment_11_pix']**2
        #sigma = self.conf['agc']['ag_pix_scale'] * varia**0.25

        fwhm = sigma * 2.355

        df = pd.DataFrame(
            data={'pfs_visit_id': pfs_visit_id,
                  'agc_exposure_id': agc_exposure_id,
                  'agc_camera_id': agc_camera_id,
                  'taken_at': taken_at,
                  'sigma_a': sigma_a,
                  'sigma_b': sigma_b,
                  'sigma': sigma,
                  'fwhm': fwhm,
                  'flags': flags,
                  })
        if self.df_seeing is None:
            self.df_seeing = df.copy()
        else:
            self.df_seeing = pd.concat([self.df_seeing, df], ignore_index=True)
        
        # calculate typical values per agc_exposure_id
        taken_at_seq = []
        agc_exposure_seq = []
        visit_seq = []
        fwhm_mean = []            # calculate mean per each AG exposure
        fwhm_median = []          # calculate median per each AG exposure
        fwhm_stddev = []          # calculate stddev per each AG exposure
        for s in seq:
            taken_at_seq.append(taken_at[agc_exposure_id == s].values[0])
            agc_exposure_seq.append(agc_exposure_id[agc_exposure_id == s].values[0])
            visit_seq.append(pfs_visit_id[agc_exposure_id == s].values[0])
            fwhm_mean.append(fwhm[agc_exposure_id == s].mean(skipna=True))
            fwhm_median.append(fwhm[agc_exposure_id == s].median(skipna=True))
            fwhm_stddev.append(fwhm[agc_exposure_id == s].std(skipna=True))
        df = pd.DataFrame(
            data={'taken_at_seq': taken_at_seq,
                  'agc_exposure_seq': agc_exposure_seq,
                  'visit_seq': visit_seq,
                  'fwhm_mean': fwhm_mean,
                  'fwhm_median': fwhm_median,
                  'fwhm_sigma': fwhm_stddev,
                  })
        if self.df_seeing_stats is None:
            self.df_seeing_stats = df.copy()
        else:
            self.df_seeing_stats = pd.concat([self.df_seeing_stats, df], ignore_index=True)

        # calculate typical values per pfs_visit_id
        visit_p_visit = []
        fwhm_mean_p_visit = []    # calculate mean per visit
        fwhm_median_p_visit = []  # calculate median per visit
        fwhm_stddev_p_visit = []  # calculate sigma per visit
        visit_p_visit.append(visit)
        dat = fwhm[pfs_visit_id == visit]
        dat_clip = dat.clip(dat.quantile(0.05), dat.quantile(0.95))
        fwhm_mean_p_visit.append(dat_clip.mean(skipna=True))
        fwhm_median_p_visit.append(dat_clip.median(skipna=True))
        fwhm_stddev_p_visit.append(dat_clip.std(skipna=True))

        # insert into qaDB
        df = pd.DataFrame(
            data={'pfs_visit_id': visit_p_visit,
                  # 'pfs_visit_description': [pfs_visit_description],
                  'pfs_design_id': [pfs_design_id],
                  # 'issued_at': [issued_at],
                  }
            )
        self.qadb.populateQATable('pfs_visit', df)

        df = pd.DataFrame(
            data={'pfs_visit_id': visit_p_visit,
                  'seeing_mean': fwhm_mean_p_visit,
                  'seeing_median': fwhm_median_p_visit,
                  'seeing_sigma': fwhm_stddev_p_visit,
                  'wavelength_ref': [self.skyQaConf["ref_wav_sky"] for _ in visit_p_visit],
                  }
            )
        df = df.fillna(-1).astype(float)
        self.qadb.populateQATable('seeing', df)
        if self.df_seeing_stats_pv is None:
            self.df_seeing_stats_pv = df.copy()
        else:
            self.df_seeing_stats_pv = pd.concat([self.df_seeing_stats_pv, df], ignore_index=True)
        
    def calcTransparency(self, visit):
        """Calculate Transparency based on AGC measurements

        Parameters
        ----------
            visit : pfs_visit_id

        Returns
        ----------
            
        Examples
        ----------

        """
        logger.info(f"calculating sky transparency for visit={visit}...")

        df = self.getAgcData(visit)

        agc_exposure_id = df['agc_exposure_id']
        seq = np.unique(agc_exposure_id)
        taken_at = df['taken_at']
        pfs_visit_id = df['pfs_visit_id'].astype(int)
        agc_camera_id = df['agc_camera_id']
        flags = df['flags']

        mag1 = df['guide_star_magnitude']
        mag2 = df['estimated_magnitude']
        dmag = mag2 - mag1
        transp = 10**(-0.4*dmag) / self.conf['agc']['transparency_correction']
        transp[transp>2.0] = 2.0

        df = pd.DataFrame(
            data={'pfs_visit_id': pfs_visit_id,
                  'agc_exposure_id': agc_exposure_id,
                  'agc_camera_id': agc_camera_id,
                  'taken_at': taken_at,
                  'guide_star_magnitude': mag1,
                  'estimated_magnitude': mag2,
                  'transp': transp,
                  'flags': flags,
                  })
        if self.df_transparency is None:
            self.df_transparency = df.copy()
        else:
            self.df_transparency = pd.concat([self.df_transparency, df], ignore_index=True)

        taken_at_seq = []
        agc_exposure_seq = []
        visit_seq = []
        transp_mean = []       # calculate mean per each AG exposure
        transp_median = []     # calculate median per each AG exposure
        transp_stddev = []     # calculate stddev per each AG exposure
        for s in seq:
            taken_at_seq.append(taken_at[agc_exposure_id == s].values[0])
            agc_exposure_seq.append(agc_exposure_id[agc_exposure_id == s].values[0])
            visit_seq.append(pfs_visit_id[agc_exposure_id == s].values[0])
            data = transp[agc_exposure_id == s]
            if len(data[data.notna()]) > 0:
                transp_mean.append(data.mean(skipna=True))
                transp_median.append(data.median(skipna=True))
                transp_stddev.append(data.std(skipna=True))
            else:
                transp_mean.append(np.nan)
                transp_median.append(np.nan)
                transp_stddev.append(np.nan)
        df = pd.DataFrame(
            data={'taken_at_seq': taken_at_seq,
                  'agc_exposure_seq': agc_exposure_seq,
                  'visit_seq': visit_seq,
                  'transp_mean': transp_mean,
                  'transp_median': transp_median,
                  'transp_sigma': transp_stddev,
                  })
        if self.df_transparency_stats is None:
            self.df_transparency_stats = df.copy()
        else:
            self.df_transparency_stats = pd.concat([self.df_transparency_stats, df], ignore_index=True)
                
        visit_p_visit = []
        transp_mean_p_visit = []    # calculate mean per visit
        transp_median_p_visit = []  # calculate median per visit
        transp_stddev_p_visit = []  # calculate sigma per visit
        for v in np.unique(pfs_visit_id):
            visit_p_visit.append(int(v))
            data = transp[pfs_visit_id == v]
            if len(data[data.notna()]) > 0:
                dat = data[pfs_visit_id == visit]
                dat_clip = dat.clip(dat.quantile(0.05), dat.quantile(0.95))
                transp_mean_p_visit.append(dat_clip.mean(skipna=True))
                transp_median_p_visit.append(dat_clip.median(skipna=True))
                transp_stddev_p_visit.append(dat_clip.std(skipna=True))
            else:
                transp_mean_p_visit.append(np.nan)
                transp_median_p_visit.append(np.nan)
                transp_stddev_p_visit.append(np.nan)
                
        # insert into qaDB
        df = pd.DataFrame(
            data={'pfs_visit_id': visit_p_visit,
                  'transparency_mean': transp_mean_p_visit,
                  'transparency_median': transp_median_p_visit,
                  'transparency_sigma': transp_stddev_p_visit,
                  'wavelength_ref': [self.skyQaConf["ref_wav_sky"] for _ in visit_p_visit],
                  }
            )
        df = df.fillna(-1).astype(float)
        self.qadb.populateQATable('transparency', df)
        if self.df_transparency_stats_pv is None:
            self.df_transparency_stats_pv = df.copy()
        else:
            self.df_transparency_stats_pv = pd.concat([self.df_transparency_stats_pv, df], ignore_index=True)


    def calcAgBackground(self,  visit):
        """Calculate Sky Backroung level based on AGC images

        Parameters
        ----------
            visit : pfs_visit_id

        Returns
        ----------
            
        Examples
        ----------

        """
        df = self.getAgcData(visit)

        agc_exposure_id = df['agc_exposure_id']
        seq = np.unique(agc_exposure_id)
        taken_at = df['taken_at']
        pfs_visit_id = df['pfs_visit_id'].astype(int)
        agc_camera_id = df['agc_camera_id']
        flags = df['flags']

        ag_background = df['background']

        df = pd.DataFrame(
            data={'pfs_visit_id': pfs_visit_id,
                  'agc_exposure_id': agc_exposure_id,
                  'agc_camera_id': agc_camera_id,
                  'taken_at': taken_at,
                  'ag_background': ag_background,
                  'flags': flags,
                  })
        if self.df_ag_background is None:
            self.df_ag_background = df.copy()
        else:
            self.df_ag_background = pd.concat([self.df_ag_background, df], ignore_index=True)

        taken_at_seq = []
        agc_exposure_seq = []
        visit_seq = []
        ag_background_mean = []       # calculate mean per each AG exposure
        ag_background_median = []     # calculate median per each AG exposure
        ag_background_stddev = []     # calculate stddev per each AG exposure
        for s in seq:
            taken_at_seq.append(taken_at[agc_exposure_id == s].values[0])
            agc_exposure_seq.append(agc_exposure_id[agc_exposure_id == s].values[0])
            visit_seq.append(pfs_visit_id[agc_exposure_id == s].values[0])
            data = ag_background[agc_exposure_id == s]
            if len(data[data.notna()]) > 0:
                ag_background_mean.append(data.mean(skipna=True))
                ag_background_median.append(data.median(skipna=True))
                ag_background_stddev.append(data.std(skipna=True))
            else:
                ag_background_mean.append(np.nan)
                ag_background_median.append(np.nan)
                ag_background_stddev.append(np.nan)
        df = pd.DataFrame(
            data={'taken_at_seq': taken_at_seq,
                  'agc_exposure_seq': agc_exposure_seq,
                  'visit_seq': visit_seq,
                  'ag_background_mean': ag_background_mean,
                  'ag_background_median': ag_background_median,
                  'ag_background_sigma': ag_background_stddev,
                  })
        if self.df_ag_background_stats is None:
            self.df_ag_background_stats = df.copy()
        else:
            self.df_ag_background_stats = pd.concat([self.df_ag_background_stats, df], ignore_index=True)
                
        visit_p_visit = []
        ag_background_mean_p_visit = []    # calculate mean per visit
        ag_background_median_p_visit = []  # calculate median per visit
        ag_background_stddev_p_visit = []  # calculate sigma per visit
        for v in np.unique(pfs_visit_id):
            visit_p_visit.append(int(v))
            data = ag_background[pfs_visit_id == v]
            if len(data[data.notna()]) > 0:
                dat = data[pfs_visit_id == visit]
                dat_clip = dat.clip(dat.quantile(0.05), dat.quantile(0.95))
                ag_background_mean_p_visit.append(dat_clip.mean(skipna=True))
                ag_background_median_p_visit.append(dat_clip.median(skipna=True))
                ag_background_stddev_p_visit.append(dat_clip.std(skipna=True))
            else:
                ag_background_mean_p_visit.append(np.nan)
                ag_background_median_p_visit.append(np.nan)
                ag_background_stddev_p_visit.append(np.nan)
                
        df = pd.DataFrame(
            data={'pfs_visit_id': visit_p_visit,
                  'ag_background_mean': ag_background_mean_p_visit,
                  'ag_background_median': ag_background_median_p_visit,
                  'ag_background_sigma': ag_background_stddev_p_visit
                  }
            )
        df = df.fillna(-1).astype(float)
        self.df_ag_background_stats_pv = df.copy()
        #if self.df_ag_background_stats_pv is None:
        #    self.df_ag_background_stats_pv = df.copy()
        #else:
        #    self.df_ag_background_stats_pv = pd.concat([self.df_ag_background_stats_pv, df], ignore_index=True)


    def calcSkyBackground(self, visit, usePfsMerged=True, usePfsSingle=False):
        """Calculate Sky background level based on SKY fibers flux and AG background level

        Parameters
        ----------
            visit : pfs_visit_id
            
        Returns
        ----------
            
        Examples
        ----------

        """
        # setup butler
        _drpDataConf = self.conf["drp"]["data"]
        baseDir = _drpDataConf["base"]
        pfsConfigDir = _drpDataConf["pfsConfigDir"]
        rerun = os.path.join(baseDir, 'rerun', _drpDataConf["rerun"])
        calibRoot = os.path.join(baseDir, _drpDataConf["calib"])
        try:
            import lsst.daf.persistence as dafPersist
            butler = dafPersist.Butler(rerun, calibRoot=calibRoot)
        except:
            butler = None
          
        # get visit information
        self.df = self.getSpsExposure(visit=visit)
        pfs_visit_id = self.df['pfs_visit_id'].astype(int)
        sps_camera_ids = self.df['sps_camera_id']
        exptimes = self.df['exptime']
        obstime = self.df['time_exp_end'][0]
        obstime = obstime.tz_localize('US/Hawaii')
        obstime = obstime.tz_convert('UTC')
        obsdate = obstime.date().strftime('%Y-%m-%d')
        _df = self.getPfsVisit(visit=visit)
        if len(_df) > 0:
            pfs_design_id = _df.pfs_design_id[0]
        else:
            pfs_design_id = None

        visit_p_visit = []
        sky_mean_p_visit = []    # calculate background level (mean over fibers) per visit
        sky_median_p_visit = []  # calculate background level (median over fibers) per visit
        sky_stddev_p_visit = []  # calculate background level (stddev over fibers) per visit
        noise_mean_p_visit = []  # calculate noise level (mean over fibers) per visit
        noise_median_p_visit = []  # calculate noise level (median over fibers) per visit
        noise_stddev_p_visit = []  # calculate noise level (stddev over fibers) per visit
        agc_bg_mean_p_visit = []    # calculate AGC background level (mean over fibers) per visit
        agc_bg_median_p_visit = []  # calculate AGC background level (median over fibers) per visit
        agc_bg_stddev_p_visit = []  # calculate AGC background level (stddev over fibers) per visit

        for v in np.unique(pfs_visit_id):
            if butler is not None:
                dataId = dict(visit=v, spectrograph=self.skyQaConf["ref_spectrograph_sky"], arm=self.skyQaConf["ref_arm_sky"],
                )
                pfsArm = butler.get("pfsArm", dataId)
                pfsMerged = butler.get("pfsMerged", dataId)
                pfsConfig = butler.get("pfsConfig", dataId)
            else:
                _pfsArmDataDir = os.path.join(rerun, 'pfsArm', obsdate, 'v%06d' % (v))
                _pfsMergedDataDir = os.path.join(rerun, 'pfsMerged', obsdate, 'v%06d' % (v))
                _pfsConfigDataDir = os.path.join(pfsConfigDir, obsdate)
                if os.path.isdir(_pfsArmDataDir) is False:
                    _pfsArmDataDir = os.path.join(rerun, 'pfsArm')
                if os.path.isdir(_pfsMergedDataDir) is False:
                    _pfsMergedDataDir = os.path.join(rerun, 'pfsMerged')
                if os.path.isdir(_pfsConfigDataDir) is False:
                    _pfsConfigDataDir = os.path.join(rerun, 'pfsConfig')
                _identity = Identity(visit=v, 
                                     arm=self.skyQaConf["ref_arm_sky"],
                                     spectrograph=self.skyQaConf["ref_spectrograph_sky"]
                                     )
                try:
                    pfsArm = PfsArm.read(_identity, dirName=_pfsArmDataDir)
                    _pfsDesignId = pfs_design_id
                    pfsMerged = PfsMerged.read(_identity, dirName=_pfsMergedDataDir)
                    pfsConfig = PfsConfig.read(pfsDesignId=_pfsDesignId, visit=v, 
                                            dirName=_pfsConfigDataDir)
                except:
                    pfsArm = None
                    pfsMerged = None
                    _pfsDesignId = None
                    pfsConfig = None
            # get AG background level
            self.calcAgBackground(visit=v)

            # calculate the typical sky background level
            logger.info(f"calculating sky background level for visit={v}...")
            if usePfsMerged is True:
                logger.info(f"pfsMerged is used")
                if pfsMerged is not None:
                    spectraSky = pfsMerged.select(pfsConfig=pfsConfig, targetType=TargetType.SKY)
                    spectraFluxstd = pfsMerged.select(pfsConfig=pfsConfig, targetType=TargetType.FLUXSTD)
                    spectraScience = pfsMerged.select(pfsConfig=pfsConfig, targetType=TargetType.SCIENCE)
                else:
                    spectraSky = None
                    spectraFluxstd = None
                    spectraScience = None
            else:
                logger.info(f"pfsArm is used")
                if pfsArm is not None:
                    spectraSky = pfsArm.select(pfsConfig=pfsConfig, targetType=TargetType.SKY)
                    spectraFluxstd = pfsArm.select(pfsConfig=pfsConfig, targetType=TargetType.FLUXSTD)
                    spectraScience = pfsArm.select(pfsConfig=pfsConfig, targetType=TargetType.SCIENCE)
                else:
                    spectraSky = None
                    spectraFluxstd = None
                    spectraScience = None
           
            if spectraSky is not None:
                data1 = []
                data2 = []
                for wav, flx, sky in zip(spectraSky.wavelength, spectraSky.flux, spectraSky.sky):
                    wc = self.skyQaConf["ref_wav_sky"]
                    dw = self.skyQaConf["ref_dwav_sky"]
                    msk = (wav > wc-dw) * (wav < wc+dw) # FIXME (SKYLINE should be masked)
                    data1.append(np.nanmedian(sky[msk]))
                    data2.append(np.nanstd(flx[msk]))
                logger.info(f"{len(data1)} SKYs are used to calculate")
                data2f = []
                for wav, flx in zip(spectraFluxstd.wavelength, spectraFluxstd.flux):
                    wc = self.skyQaConf["ref_wav_sky"]
                    dw = self.skyQaConf["ref_dwav_sky"]
                    msk = (wav > wc-dw) * (wav < wc+dw) # FIXME (SKYLINE should be masked)
                    data2f.append(np.nanstd(flx[msk]))
                data2s = []
                for wav, flx in zip(spectraScience.wavelength, spectraScience.flux):
                    wc = self.skyQaConf["ref_wav_sky"]
                    dw = self.skyQaConf["ref_dwav_sky"]
                    msk = (wav > wc-dw) * (wav < wc+dw) # FIXME (SKYLINE should be masked)
                    data2s.append(np.nanstd(flx[msk]))
                # store info into dataframe
                df1 = pd.DataFrame(
                    data={'pfs_visit_id': [v for _ in range(len(spectraSky))],
                        'fiber_id': spectraSky.fiberId,
                        'background_level': data1,
                        }
                    )
                if self.df_sky_background is None:
                    self.df_sky_background = df1.copy()
                else:
                    self.df_sky_background = pd.concat([self.df_sky_background, df1.copy()], ignore_index=True)

                df2 = pd.DataFrame(
                    data={'pfs_visit_id': [v for _ in range(len(spectraSky))],
                        'fiber_id': spectraSky.fiberId,
                        'noise_level': data2,
                        }
                    )
                df2f = pd.DataFrame(
                    data={'pfs_visit_id': [v for _ in range(len(spectraFluxstd))],
                        'fiber_id': spectraFluxstd.fiberId,
                        'noise_level': data2f,
                        }
                    )
                df2s = pd.DataFrame(
                    data={'pfs_visit_id': [v for _ in range(len(spectraScience))],
                        'fiber_id': spectraScience.fiberId,
                        'noise_level': data2s,
                        }
                    )
                if self.df_sky_noise is None:
                    self.df_sky_noise = df2.copy()
                else:
                    self.df_sky_noise = pd.concat([self.df_sky_noise, df2.copy()], ignore_index=True)
                visit_p_visit.append(v)
                if self.df_flxstd_noise is None:
                    self.df_flxstd_noise = df2f.copy()
                else:
                    self.df_flxstd_noise = pd.concat([self.df_flxstd_noise, df2f.copy()], ignore_index=True)
                if self.df_science_noise is None:
                    self.df_science_noise = df2s.copy()
                else:
                    self.df_science_noise = pd.concat([self.df_science_noise, df2s.copy()], ignore_index=True)
                #sky = df1['background_level']
                dat = df1.query(f'pfs_visit_id=={visit}').background_level
                dat_clip = dat.clip(dat.quantile(0.05), dat.quantile(0.95))
                sky_mean_p_visit.append(dat_clip.mean(skipna=True))
                sky_median_p_visit.append(dat_clip.median(skipna=True))
                sky_stddev_p_visit.append(dat_clip.std(skipna=True))
                #noise = df2['noise_level']
                dat = df2.query(f'pfs_visit_id=={visit}').noise_level
                dat_clip = dat.clip(dat.quantile(0.05), dat.quantile(0.95))
                noise_mean_p_visit.append(dat_clip.mean(skipna=True))
                noise_median_p_visit.append(dat_clip.median(skipna=True))
                noise_stddev_p_visit.append(dat_clip.std(skipna=True))
            else:
                logger.info(f'visit={v} skipped...')

        # populate database (sky table)
        df1 = pd.DataFrame(
            data={'pfs_visit_id': visit_p_visit,
                  'sky_background_mean': sky_mean_p_visit,
                  'sky_background_median': sky_median_p_visit,
                  'sky_background_sigma': sky_stddev_p_visit,
                  'wavelength_ref': [self.skyQaConf["ref_wav_sky"] for _ in visit_p_visit],
                  'agc_background_mean': self.df_ag_background_stats_pv['ag_background_mean'].values,
                  'agc_background_median': self.df_ag_background_stats_pv['ag_background_median'].values,
                  'agc_background_sigma': self.df_ag_background_stats_pv['ag_background_sigma'].values,
                  }
            )
        self.qadb.populateQATable('sky', df1)
        if self.df_sky_background_stats_pv is None:
            self.df_sky_background_stats_pv = df1.copy()
        else:
            self.df_sky_background_stats_pv = pd.concat([self.df_sky_background_stats_pv, df1.copy()], ignore_index=True)

        # populate database (noise table)
        df2 = pd.DataFrame(
            data={'pfs_visit_id': visit_p_visit,
                  'noise_mean': noise_mean_p_visit,
                  'noise_median': noise_median_p_visit,
                  'noise_sigma': noise_stddev_p_visit,
                  'wavelength_ref': [self.skyQaConf["ref_wav_sky"] for _ in visit_p_visit],
                  }
            )
        self.qadb.populateQATable('noise', df2)
        if self.df_sky_noise_stats_pv is None:
            self.df_sky_noise_stats_pv = df2.copy()
        else:
            self.df_sky_noise_stats_pv = pd.concat([self.df_sky_noise_stats_pv, df2.copy()], ignore_index=True)

    def calcThroughput(self, visit, usePfsMerged=True, usePfsFluxReference=True):
        """Calculate total throughput based on FLUXSTD fibers flux

        Parameters
        ----------
            visit : pfs_visit_id
            
        Returns
        ----------
            
        Examples
        ----------

        """
        # setup butler
        _drpDataConf = self.conf["drp"]["data"]
        baseDir = _drpDataConf["base"]
        pfsConfigDir = _drpDataConf["pfsConfigDir"]
        rerun = os.path.join(baseDir, 'rerun', _drpDataConf["rerun"])
        calibRoot = os.path.join(baseDir, _drpDataConf["calib"])
        try:
            import lsst.daf.persistence as dafPersist
            butler = dafPersist.Butler(rerun, calibRoot=calibRoot)
        except:
            butler = None
          
        # get visit information
        self.df = self.getSpsExposure(visit=visit)
        pfs_visit_id = self.df['pfs_visit_id'].astype(int)
        sps_camera_ids = self.df['sps_camera_id']
        exptimes = self.df['exptime']
        obstime = self.df['time_exp_end'][0]
        obstime = obstime.tz_localize('US/Hawaii')
        obstime = obstime.tz_convert('UTC')
        obsdate = obstime.date().strftime('%Y-%m-%d')
        _df = self.getPfsVisit(visit=visit)
        if len(_df) > 0:
            pfs_design_id = _df.pfs_design_id[0]
        else:
            pfs_design_id = None

        visit_p_visit = []
        throughput_mean_p_visit = []    # calculate throughput (mean over fibers) per visit
        throughput_median_p_visit = []  # calculate throughput (median over fibers) per visit
        throughput_stddev_p_visit = []  # calculate throughput (stddev over fibers) per visit
        for v in np.unique(pfs_visit_id):
            if butler is not None:
                dataId = dict(visit=v, spectrograph=self.skyQaConf["ref_spectrograph_sky"], arm=self.skyQaConf["ref_arm_sky"],
                )
                pfsArm = butler.get("pfsArm", dataId)
                pfsMerged = butler.get("pfsMerged", dataId)
                pfsConfig = butler.get("pfsConfig", dataId)
                if usePfsFluxReference is True:
                    pfsFluxReference = butler.get("pfsFluxReference", dataId)
                else:
                    pfsFluxReference = None
            else:
                _pfsArmDataDir = os.path.join(rerun, 'pfsArm', obsdate, 'v%06d' % (v))
                _pfsMergedDataDir = os.path.join(rerun, 'pfsMerged', obsdate, 'v%06d' % (v))
                _pfsFluxReferenceDataDir = os.path.join(rerun, 'pfsFluxReference', obsdate, 'v%06d' % (v))
                _pfsConfigDataDir = os.path.join(pfsConfigDir, obsdate)
                if os.path.isdir(_pfsArmDataDir) is False:
                    _pfsArmDataDir = os.path.join(rerun, 'pfsArm')
                if os.path.isdir(_pfsMergedDataDir) is False:
                    _pfsMergedDataDir = os.path.join(rerun, 'pfsMerged')
                if os.path.isdir(_pfsFluxReferenceDataDir) is False:
                    _pfsFluxReferenceDataDir = os.path.join(rerun, 'pfsFluxReference')
                if os.path.isdir(_pfsConfigDataDir) is False:
                    _pfsConfigDataDir = os.path.join(rerun, 'pfsConfig')
                _identity = Identity(visit=v, 
                                     arm=self.skyQaConf["ref_arm_sky"],
                                     spectrograph=self.skyQaConf["ref_spectrograph_sky"]
                                     )
                try:
                    pfsArm = PfsArm.read(_identity, dirName=_pfsArmDataDir)
                    _pfsDesignId = pfs_design_id
                    pfsMerged = PfsMerged.read(_identity, dirName=_pfsMergedDataDir)
                    pfsConfig = PfsConfig.read(pfsDesignId=_pfsDesignId, visit=v, 
                                            dirName=_pfsConfigDataDir)
                    if usePfsFluxReference is True:
                        pfsFluxReference = PfsFluxReference.read(_identity, dirName=_pfsFluxReferenceDataDir)
                    else:
                        pfsFluxReference = None
                except:
                    pfsArm = None
                    pfsMerged = None
                    _pfsDesignId = None
                    pfsConfig = None
                    pfsFluxReference = None
            # calculate the throughput from FLUXSTD spectra
            logger.info(f"calculating throughput for visit={v}...")

            ref_wav = self.skyQaConf["ref_wav_sky"]
            if ref_wav < 550.:
                idx_psfFlux = 0
            elif ref_wav < 700.:
                idx_psfFlux = 1
            elif ref_wav < 800.:
                idx_psfFlux = 2
            elif ref_wav < 900.:
                idx_psfFlux = 3
            else:
                idx_psfFlux = 4
            
            if usePfsMerged is True:
                logger.info(f"pfsMerged is used")
                if pfsMerged is not None:
                    SpectraFluxstd = pfsMerged.select(pfsConfig=pfsConfig, targetType=TargetType.FLUXSTD)
                else:
                    SpectraFluxstd = None
            else:
                logger.info(f"pfsArm is used")
                if pfsArm is not None:
                    SpectraFluxstd = pfsArm.select(pfsConfig=pfsConfig, targetType=TargetType.FLUXSTD)
                else:
                    SpectraFluxstd = None

            if SpectraFluxstd is not None:                   
                throughput = []
                for fid, wav, flx in zip(SpectraFluxstd.fiberId, SpectraFluxstd.wavelength, SpectraFluxstd.flux):
                    wc = self.skyQaConf["ref_wav_sky"]
                    dw = self.skyQaConf["ref_dwav_sky"]
                    if usePfsFluxReference is True:
                        wav_pfr = np.array(pfsFluxReference.wavelength.tolist())
                        flx_pfr = pfsFluxReference.flux[pfsFluxReference.fiberId==fid][0]
                        msk = (wav_pfr > wc - dw) * (wav_pfr < wc + dw) # FIXME?
                        refFlux = np.nanmedian(flx_pfr[msk])
                    else:
                        refFlux = pfsConfig[pfsConfig.fiberId==fid].psfFlux[0][idx_psfFlux]

                    msk = (wav > wc-dw) * (wav < wc+dw) # FIXME (SKYLINE should be masked)
                    throughput.append(np.nanmedian(flx[msk]) / refFlux * 5)
                logger.info(f"{len(throughput)} FLUXSTDs are used to calculate")
                visit_p_visit.append(v)
                if len(throughput) > 0:
                    throughput_mean_p_visit.append(np.nanmean(throughput))
                    throughput_median_p_visit.append(np.nanmedian(throughput))
                    throughput_stddev_p_visit.append(np.nanstd(throughput))
                else:
                    throughput_mean_p_visit.append(-1)
                    throughput_median_p_visit.append(-1)
                    throughput_stddev_p_visit.append(-1)
                # store info into dataframe
                df = pd.DataFrame(
                    data={'pfs_visit_id': [v for _ in range(len(SpectraFluxstd))],
                        'fiber_id': SpectraFluxstd.fiberId,
                        'throughput': throughput,
                        }
                    )
                if self.df_throughput is None:
                    self.df_throughput = df.copy()
                else:
                    self.df_throughput = pd.concat([self.df_throughput, df], ignore_index=True)
            else:
                logger.info(f'visit={v} skipped...')

        # populate database (throughput table)
        df = pd.DataFrame(
            data={'pfs_visit_id': visit_p_visit,
                  'throughput_mean': throughput_mean_p_visit,
                  'throughput_median': throughput_median_p_visit,
                  'throughput_sigma': throughput_stddev_p_visit,
                  'wavelength_ref': [self.skyQaConf["ref_wav_sky"] for _ in visit_p_visit],
                  }
            )
        self.qadb.populateQATable('throughput', df)
        if self.df_throughput_stats_pv is None:
            self.df_throughput_stats_pv = df.copy()
        else:
            self.df_throughput_stats_pv = pd.concat([self.df_throughput_stats_pv, df], ignore_index=True)

    def plotSeeing(self, cc='cameraId', xaxis='taken_at', saveFig=False, figName='', dirName=None):
        if self.df_seeing is not None:
            if len(self.df_seeing) > 0:
                fwhm = self.df_seeing['fwhm']
                fig = plt.figure(figsize=(8, 5))
                axe = fig.add_subplot()
                axe.set_ylabel('seeing FWHM (arcsec.)')
                axe.set_title(f'visits:{min(self.visitList)}..{max(self.visitList)}')
                axe.set_ylim(0., 2.0)
                if xaxis=='taken_at':
                    axe.set_xlabel('taken_at (HST)')
                    x_data = self.df_seeing['taken_at']
                    x_data_stats = self.df_seeing_stats['taken_at_seq']
                elif xaxis=='agc_exposure_id':
                    axe.set_xlabel('agc_exposure_id')
                    x_data = self.df_seeing['agc_exposure_id']
                    x_data_stats = self.df_seeing_stats['agc_exposure_seq']
                elif xaxis=='pfs_visit_id':
                    axe.set_xlabel('pfs_visit_id')
                    x_data = self.df_seeing['pfs_visit_id']
                    x_data_stats = self.df_seeing_stats['visit_seq']
                if cc == 'cameraId':
                    agc_camera_id = self.df_seeing['agc_camera_id']
                    for cid in np.unique(agc_camera_id):
                        msk = agc_camera_id == cid
                        axe.scatter(x_data[msk], fwhm[msk], marker='o', s=10, alpha=0.5, rasterized=True, label=f'cameraId={cid}')
                elif cc == 'flags':
                    flags = self.df_seeing['flags']
                    for cid in np.unique(flags):
                        msk = flags == cid
                        axe.scatter(x_data[msk], fwhm[msk], marker='o', s=10, alpha=0.5, rasterized=True, label=f'flags={cid}')
                elif cc == 'visit':
                    for v in self.visitList:
                        msk = self.df_seeing['pfs_visit_id'] == v
                        axe.scatter(x_data[msk], fwhm[msk], marker='o', s=10, alpha=0.5, rasterized=True, label=f'visit={v}')
                else:
                    axe.scatter(x_data, fwhm, marker='o', s=10, edgecolor='none', facecolor='C0', alpha=0.5, rasterized=True, label=f'all')
                axe.plot(x_data_stats, self.df_seeing_stats['fwhm_mean'], 
                         ls='solid', lw=2, color='k', alpha=0.8)
                axe.plot([min(x_data_stats), max(x_data_stats)],
                        [0.8, 0.8], ls='dashed', lw=1, color='k', alpha=0.8)
                axe.legend(loc='upper left', ncol=2, fontsize=8)
                plt.show()
                if saveFig == True:
                    plt.savefig(os.path.join(dirName, f'{figName}_seeing.pdf'), bbox_inches='tight')


    def plotTransparency(self, cc='cameraId', xaxis='taken_at', saveFig=False, figName='', dirName=None):
        if self.df_transparency is not None:
            if len(self.df_transparency) > 0:
                transp = self.df_transparency['transp']
                fig = plt.figure(figsize=(8, 5))
                axe = fig.add_subplot()
                axe.set_xlabel('taken_at (HST)')
                axe.set_ylabel(r'transparency')
                axe.set_title(f'visits:{min(self.visitList)}..{max(self.visitList)}')
                axe.set_ylim(0., 2.0)
                if xaxis=='taken_at':
                    axe.set_xlabel('taken_at (HST)')
                    x_data = self.df_transparency['taken_at']
                    x_data_stats = self.df_transparency_stats['taken_at_seq']
                elif xaxis=='agc_exposure_id':
                    axe.set_xlabel('agc_exposure_id')
                    x_data = self.df_transparency['agc_exposure_id']
                    x_data_stats = self.df_transparency_stats['agc_exposure_seq']
                elif xaxis=='pfs_visit_id':
                    axe.set_xlabel('pfs_visit_id')
                    x_data = self.df_transparency['pfs_visit_id']
                    x_data_stats = self.df_transparency_stats['visit_seq']
                if cc == 'cameraId':
                    agc_camera_id = self.df_transparency['agc_camera_id']
                    for cid in np.unique(agc_camera_id):
                        msk = agc_camera_id == cid
                        axe.scatter(x_data[msk], transp[msk], marker='o', s=10, alpha=0.5, rasterized=True, label=f'cameraId={cid}')
                elif cc == 'flags':
                    # flags: 0 (left) 1 (right)
                    flags = self.df_transparency['flags']
                    for cid in np.unique(flags):
                        msk = flags == cid
                        axe.scatter(x_data[msk], transp[msk], marker='o', s=10, alpha=0.5, rasterized=True, label=f'flags={cid}')
                elif cc == 'visit':
                    for v in self.visitList:
                        msk = self.df_transparency['pfs_visit_id'] == v
                        axe.scatter(x_data[msk], transp[msk], marker='o', s=10, alpha=0.5, rasterized=True, label=f'visit={v}')
                else:
                    axe.scatter(x_data, transp, marker='o', s=10, edgecolor='none', facecolor='C0', alpha=0.5, rasterized=True, label='all')
                axe.plot(x_data_stats, 
                         self.df_transparency_stats['transp_mean'], 
                         ls='solid', lw=2, color='k', alpha=0.8)
                axe.plot([min(x_data_stats), max(x_data_stats)],
                        [1.0, 1.0], ls='dashed', lw=1, color='k', alpha=0.8)
                axe.legend(loc='upper left', ncol=2, fontsize=8)
                plt.show()
                if saveFig == True:
                    plt.savefig(os.path.join(dirName, f'{figName}_transparency.pdf'), bbox_inches='tight')

    def plotAgBackground(self, cc='cameraId', xaxis='taken_at', saveFig=False, figName='', dirName=None):
        if self.df_ag_background is not None:
            if len(self.df_ag_background) > 0:
                ag_background = self.df_ag_background['ag_background']
                fig = plt.figure(figsize=(8, 5))
                axe = fig.add_subplot()
                axe.set_xlabel('taken_at (HST)')
                axe.set_ylabel(r'AG background')
                axe.set_title(f'visits:{min(self.visitList)}..{max(self.visitList)}')
                #axe.set_ylim(0., 2.0)
                if xaxis=='taken_at':
                    axe.set_xlabel('taken_at (HST)')
                    x_data = self.df_ag_background['taken_at']
                    x_data_stats = self.df_ag_background_stats['taken_at_seq']
                elif xaxis=='agc_exposure_id':
                    axe.set_xlabel('agc_exposure_id')
                    x_data = self.df_ag_background['agc_exposure_id']
                    x_data_stats = self.df_ag_background_stats['agc_exposure_seq']
                elif xaxis=='pfs_visit_id':
                    axe.set_xlabel('pfs_visit_id')
                    x_data = self.df_ag_background['pfs_visit_id']
                    x_data_stats = self.df_ag_background_stats['visit_seq']
                if cc == 'cameraId':
                    agc_camera_id = self.df_ag_background['agc_camera_id']
                    for cid in np.unique(agc_camera_id):
                        msk = agc_camera_id == cid
                        axe.scatter(x_data[msk], ag_background[msk], marker='o', s=10, alpha=0.5, rasterized=True, label=f'cameraId={cid}')
                elif cc == 'flags':
                    # flags: 0 (left) 1 (right)
                    flags = self.df_ag_background['flags']
                    for cid in np.unique(flags):
                        msk = flags == cid
                        axe.scatter(x_data[msk], ag_background[msk], marker='o', s=10, alpha=0.5, rasterized=True, label=f'flags={cid}')
                elif cc == 'visit':
                    for v in self.visitList:
                        msk = self.df_ag_background['pfs_visit_id'] == v
                        axe.scatter(x_data[msk], ag_background[msk], marker='o', s=10, alpha=0.5, rasterized=True, label=f'visit={v}')
                else:
                    axe.scatter(x_data, ag_background, marker='o', s=10, edgecolor='none', facecolor='C0', alpha=0.5, rasterized=True, label='all')
                axe.plot(x_data_stats, 
                         self.df_ag_background_stats['ag_background_mean'], 
                         ls='solid', lw=2, color='k', alpha=0.8)
                # axe.plot([min(x_data_stats), max(x_data_stats)], [1.0, 1.0], ls='dashed', lw=1, color='k', alpha=0.8)
                axe.legend(loc='upper left', ncol=2, fontsize=8)
                plt.show()
                if saveFig == True:
                    plt.savefig(os.path.join(dirName, f'{figName}_ag_background.pdf'), bbox_inches='tight')

    def plotSkyBackground(self, saveFig=False, figName='', dirName=None):
        if self.df_sky_background_stats_pv is not None:
            if len(self.df_sky_background_stats_pv) > 0:
                sky_background_visits = self.df_sky_background_stats_pv['pfs_visit_id']
                sky_background = self.df_sky_background_stats_pv['sky_background_median']
                sky_noise_visits = self.df_sky_noise_stats_pv['pfs_visit_id']
                sky_noise = self.df_sky_noise_stats_pv['noise_median']
                fig = plt.figure(figsize=(8, 5))
                ax1 = fig.add_subplot(211)
                ax1.set_ylabel(r'Sky background level')
                ax1.set_title(f'visits:{min(sky_background_visits)}..{max(sky_background_visits)}')
                #ax1.scatter(sky_background_visits, sky_background, marker='o', s=10, alpha=0.5, rasterized=True)
                ax1.plot(sky_background_visits, sky_background, ls='solid', lw=2, color='k', alpha=0.8)
                ax2 = fig.add_subplot(212)
                ax2.set_xlabel('pfs_visit_id')
                ax2.set_ylabel(r'Sky background noise')
                #ax2.scatter(sky_noise_visits, sky_noise, marker='o', s=10, alpha=0.5, rasterized=True)
                ax2.plot(sky_noise_visits, sky_noise, ls='solid', lw=2, color='k', alpha=0.8)                
                plt.show()
                if saveFig == True:
                    plt.savefig(os.path.join(dirName, f'{figName}_sky_background.pdf'), bbox_inches='tight')

    def plotThroughput(self, saveFig=False, figName='', dirName=None):
        if self.df_throughput_stats_pv is not None:
            if len(self.df_throughput_stats_pv) > 0:
                throughput_visits = self.df_throughput_stats_pv['pfs_visit_id']
                throughputs = self.df_throughput_stats_pv['throughput_median']
                fig = plt.figure(figsize=(8, 5))
                axe = fig.add_subplot()
                axe.set_xlabel(r'pfs_visit_id')
                axe.set_ylabel(r'Throughput')
                axe.set_title(f'visits:{min(throughput_visits)}..{max(throughput_visits)}')
                #axe.scatter(throughput_visits, throughputs, marker='o', s=10, alpha=0.5, rasterized=True)
                axe.plot(throughput_visits, 
                         throughputs,
                         ls='solid', lw=2, color='k', alpha=0.8)
                plt.show()
                if saveFig == True:
                    plt.savefig(os.path.join(dirName, f'{figName}_throughput.pdf'), bbox_inches='tight')

    def plotGuideError(self, cc='cameraId', xaxis='taken_at', saveFig=False, figName='', dirName=None):
        if self.df_guide_error is not None:
            if len(self.df_guide_error) > 0:
                taken_at = self.df_guide_error['taken_at']
                agc_exposure_id = self.df_guide_error['agc_exposure_id']
                pfs_visit_id = self.df_guide_error['pfs_visit_id']
                guide_delta_ra = self.df_guide_error['guide_delta_ra']
                guide_delta_dec = self.df_guide_error['guide_delta_dec']
                guide_delta_insrot = self.df_guide_error['guide_delta_insrot']
                guide_delta_az = self.df_guide_error['guide_delta_az']
                guide_delta_el = self.df_guide_error['guide_delta_el']
                guide_delta_z = self.df_guide_error['guide_delta_z'] * 1000
                guide_delta_z1 = self.df_guide_error['guide_delta_z1'] * 1000
                guide_delta_z2 = self.df_guide_error['guide_delta_z2'] * 1000
                guide_delta_z3 = self.df_guide_error['guide_delta_z3'] * 1000
                guide_delta_z4 = self.df_guide_error['guide_delta_z4'] * 1000
                guide_delta_z5 = self.df_guide_error['guide_delta_z5'] * 1000
                guide_delta_z6 = self.df_guide_error['guide_delta_z6'] * 1000

                fig = plt.figure(figsize=(8, 8))

                ax1 = fig.add_subplot(311)
                ax1.set_ylabel('delta_ra/dec (arcsec.)')
                ax1.set_ylim(-1., +1.)
                ax1.set_title(f'visits:{min(self.visitList)}..{max(self.visitList)}')
                if xaxis=='taken_at':
                    x_data = taken_at
                elif xaxis=='agc_exposure_id':
                    x_data = agc_exposure_id
                elif xaxis=='pfs_visit_id':
                    x_data = pfs_visit_id
                ax1.plot(x_data, guide_delta_ra, alpha=0.5, label=f'guide_delta_ra')
                ax1.plot(x_data, guide_delta_dec, alpha=0.5, label=f'guide_delta_dec')
                ax1.legend(loc='lower left', ncol=2, fontsize=8)

                ax2 = fig.add_subplot(312)
                ax2.set_ylabel('delta_rot (arcsec.)')
                ax2.set_ylim(-30., +30.)
                if xaxis=='taken_at':
                    x_data = taken_at
                elif xaxis=='agc_exposure_id':
                    x_data = agc_exposure_id
                elif xaxis=='pfs_visit_id':
                    x_data = pfs_visit_id
                ax2.plot(x_data, guide_delta_insrot, alpha=0.5, label=f'guide_delta_insrot')
                ax2.legend(loc='lower left', ncol=2, fontsize=8)

                ax3 = fig.add_subplot(313)
                ax3.set_ylabel('focus offset (micron)')
                ax3.set_ylim(-700., +700.)
                if xaxis=='taken_at':
                    ax3.set_xlabel('taken_at (HST)')
                elif xaxis=='agc_exposure_id':
                    ax3.set_xlabel('agc_exposure_id')
                elif xaxis=='pfs_visit_id':
                    ax3.set_xlabel('pfs_visit_id')
                ax3.plot(x_data, guide_delta_z, color='k', lw=3, alpha=0.5, label=f'guide_delta_z')
                ax3.plot(x_data, guide_delta_z1, alpha=0.5, label=f'guide_delta_z1')
                ax3.plot(x_data, guide_delta_z2, alpha=0.5, label=f'guide_delta_z2')
                ax3.plot(x_data, guide_delta_z1, alpha=0.5, label=f'guide_delta_z3')
                ax3.plot(x_data, guide_delta_z1, alpha=0.5, label=f'guide_delta_z4')
                ax3.plot(x_data, guide_delta_z1, alpha=0.5, label=f'guide_delta_z5')
                ax3.plot(x_data, guide_delta_z1, alpha=0.5, label=f'guide_delta_z6')
                ax3.legend(loc='upper left', ncol=3, fontsize=8)

                plt.show()
                if saveFig == True:
                    plt.savefig(os.path.join(dirName, f'{figName}_guide_error.pdf'), bbox_inches='tight')

    def getCondition(self, visits, showPlot=True, 
                     xaxis='taken_at', cc='cameraId', 
                     skipCalcSkyBackground=False, skipCalcThroughput=False,
                     saveFig=False, figName='', dirName='.'):
        """Calculate various observing condition such as seeing, transparency, etc. 

        Parameters
        ----------
            visits : `pfs_visit_id` or `list` of `pfs_visit_id`
            showPlot : `bool` (default: True)
            xaxis : `str` (default: taken_at)
            cc : `str` (default: cameraId)
            saveFig : `bool` (default: False)
            figName : `str`
            dirName : `str`

        Examples
        ----------
        """
        for visit in visits:
            print(f'visit={visit}')
            if visit not in self.visitList:
                # get seeing
                self.calcSeeing(visit=visit)
                # get transparency
                self.calcTransparency(visit=visit)
                # getGuideOffset
                self.getGuideError(visit=visit)
                # get background level
                if skipCalcSkyBackground==False:
                    self.calcSkyBackground(visit=visit)
                # get throughput
                if skipCalcThroughput==False:
                    self.calcThroughput(visit=visit)
                # append visit
                self.visitList.append(visit)
            else:
                logger.warning(f"already calculated for visit={visit}...")
                pass
        if showPlot == True:
            self.plotSeeing(cc=cc, xaxis=xaxis, saveFig=saveFig, figName=figName, dirName=dirName)
            self.plotTransparency(cc=cc, xaxis=xaxis, saveFig=saveFig, figName=figName, dirName=dirName)
            self.plotAgBackground(cc=cc, xaxis=xaxis, saveFig=saveFig, figName=figName, dirName=dirName)
            self.plotSkyBackground(saveFig=saveFig, figName=figName, dirName=dirName)
            self.plotThroughput(saveFig=saveFig, figName=figName, dirName=dirName)
            self.plotGuideError(cc=cc, xaxis=xaxis, saveFig=saveFig, figName=figName, dirName=dirName)

    def fetch_visit(self, visitStart, visitEnd=None):
        if visitEnd is None:
            sqlWhere = f'pfs_visit_id>={visitStart}'
        else:
            sqlWhere = f'pfs_visit_id>={visitStart} AND pfs_visit_id<={visitEnd}'
        sqlCmd = f'SELECT * FROM pfs_visit WHERE {sqlWhere};'
        df = pd.read_sql(sql=sqlCmd, con=self.opdb._conn)
        return df

    def auto_update(self, visitEnd=None, xaxis='taken_at', cc='cameraId'):
        df = self.fetch_visit(min(self.visitList), visitEnd)
        visits_to_send = []
        for visit in df['pfs_visit_id']:
            if visit not in self.visitList:
                visits_to_send.append(visit)
            
        self.getCondition(visits=visits_to_send, showPlot=True, xaxis=xaxis, cc=cc)

    def update(self, visit, showPlot=True, xaxis='taken_at', cc='cameraIid'):
        self.getCondition(visits=[visit], showPlot=showPlot, xaxis=xaxis, cc=cc)
        return 0
