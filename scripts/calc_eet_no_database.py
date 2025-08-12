import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import threading
import argparse

# import qa related modules
try:
    from pfs_obsproc_qa.utils.obsCondition import Condition
    from pfs_obsproc_qa.utils.exposureTime import ExposureTime
    from pfs_obsproc_qa.utils import opDB, qaDB, utils
except:
    sys.path.append("/work/kiyoyabe/erun/run22/prep/design/src/pfs_obsproc_qa_tools/src/")
    from pfs_obsproc_qa.utils.obsCondition import Condition
    from pfs_obsproc_qa.utils.exposureTime import ExposureTime
    from pfs_obsproc_qa.utils import opDB, qaDB, utils

# define some nominal values for QA (currently average values of the sample of this test)

def run(workDir, config, visits, skipAg=False, skipDrp=False, useBackground=False, updateDB=False):
    np.random.seed(1)

    figDir = os.path.join(workDir, 'figures')
    config = os.path.join(workDir, f'{config}')

    visitMin = min(visits)

    print(config)
    cond=Condition(conf=config)

    # getCondtion from AGC data
    if skipAg is False:
        cond.getConditionAg(visits=visits, showPlot=False, xaxis='taken_at', cc='flags', saveFig=True, figName='test_flags', dirName=figDir, corrColor=True, updateDB=False)

    # getCondtion from DRP products
    if skipDrp is False:
        cond.getConditionDrp(visits=visits, showPlot=False, xaxis='taken_at', cc='flags', saveFig=True, figName='test_flags', dirName=figDir, usePfsMerged=True, updateDB=False)

    # setup QA database and read some QA results (seeing, transparency, noise)

    # get effective exposure time 
    sqlWhere = f"pfs_visit_id>={visitMin}"
    conf = utils.read_conf(config)
    qadb = qaDB.QaDB(conf["db"]["qadb"])
    df_seeing = qadb.query(f'SELECT * FROM seeing WHERE {sqlWhere};')
    df_transparency = qadb.query(f'SELECT * FROM transparency WHERE {sqlWhere};')
    df_throughput = cond.df_throughput_stats_pv
    df_sky_background = cond.df_sky_background_stats_pv
    df_sky_noise = cond.df_sky_noise_stats_pv

    # setup for calculating the effective exposure time

    exp = ExposureTime(conf=config, isFaint=True)

    # calculate the effective exposure time for each visit
    for visit in visits:
        # calc nominal exposure time
        t_nominal = exp.getNominalExposureTime(visit)

        # real data from QaDB
        seeing = df_seeing.seeing_median[df_seeing.pfs_visit_id==visit].values[0]
        transparency = df_transparency.transparency_median[df_transparency.pfs_visit_id==visit].values[0]
        noise_b = df_sky_noise.noise_b_median[df_sky_noise.pfs_visit_id==visit].values[0]
        noise_r = df_sky_noise.noise_r_median[df_sky_noise.pfs_visit_id==visit].values[0]
        noise_n = df_sky_noise.noise_n_median[df_sky_noise.pfs_visit_id==visit].values[0]
        noise_m = df_sky_noise.noise_m_median[df_sky_noise.pfs_visit_id==visit].values[0]
        noises = [noise_b, noise_r, noise_n, noise_m]
        background = df_sky_background.sky_background_r_median[df_sky_background.pfs_visit_id==visit].values[0]
        backgrounds = [background, background, background, background]
        throughput_b = df_throughput.throughput_b_median[df_throughput.pfs_visit_id==visit].values[0]
        throughput_r = df_throughput.throughput_r_median[df_throughput.pfs_visit_id==visit].values[0]
        throughput_n = df_throughput.throughput_n_median[df_throughput.pfs_visit_id==visit].values[0]
        throughput_m = df_throughput.throughput_m_median[df_throughput.pfs_visit_id==visit].values[0]
        throughputs = [throughput_b, throughput_r, throughput_n, throughput_m]
        t_effectives = exp.calcEffectiveExposureTime(visit, seeing, transparency, 
                                                     throughput_b, throughput_r, throughput_n, throughput_m,
                                                     background, noise_b, noise_r, noise_n, noise_m,
                                                     useBackground=useBackground, useFluxstd=True, updateDB=False)
        
        print(f"visit={visit}:")
        print(f"    texp_nominal={t_nominal}")
        print(f"    seeing={seeing}, transparency={transparency}")
        print(f"    background={backgrounds}")
        print(f"    noise={noises}")
        print(f"    throughput={throughputs}")
        print(f"    eet={t_effectives}")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        "--skipAg",
        action="store_true",
        help="skip condition measurements from AG images (e.g., seeing, transparency)",
    )
    parser.add_argument(
        "--skipDrp",
        action="store_true",
        help="skip condition measurements from DRP products (e.g., noise, throughput)",
    )
    parser.add_argument(
        "--useBackground",
        action="store_true",
        help="use background level to calculate Effective Exposure Time",
    )
    parser.add_argument(
        "--updateDB",
        action="store_true",
        help="update qaDB",
    )
    parser.add_argument(
        "--visits",
        nargs="+",
        type=int,
        default=[],
        help="visit ID to process"
    )
    parser.add_argument(
        "--workDir",
        type=str,
        default='/work/kiyoyabe/erun/run21/get_condition',
        help="path to the work directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default='config.run21.qa.toml',
        help="configuration toml file",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="delay time to start",
    )

    args = parser.parse_args()

    time.sleep(args.delay)

    run(args.workDir, args.config, args.visits, args.skipAg, args.skipDrp, args.useBackground, args.updateDB)
    
