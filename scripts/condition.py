#! /usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import threading
import argparse 

from pfs_obsproc_qa.utils.obsCondition import Condition
from pfs_obsproc_qa.utils.exposureTime import ExposureTime
from pfs_obsproc_qa.utils import opDB, qaDB, utils


# define some nominal values for QA (currently average values of the sample of this test)

TEXP_NOMINAL = 900.0            # sec.
SEEING_NOMINAL = 0.80           # arcsec
TRANSPARENCY_NOMINAL = 0.90     # 
NOISE_LEVEL_NOMINAL = 200.0     # electron
THROUGHPUT_NOMINAL = 0.3        #


def check_opdb(db, visit0):
    sqlCmd = f"SELECT pfs_visit.pfs_visit_id from pfs_visit JOIN agc_exposure ON pfs_visit.pfs_visit_id=agc_exposure.pfs_visit_id WHERE pfs_visit.pfs_visit_id>={visit0} ORDER BY pfs_visit_id DESC;"
    df = db.query(sqlCmd)
    return np.unique(df.pfs_visit_id.to_list())

def check_qadb(db, visit0, visits):
    sqlCmd = f"SELECT * from seeing WHERE pfs_visit_id>={visit0} AND wavelength_ref>0 ORDER BY pfs_visit_id DESC;"
    df = db.query(sqlCmd)
    visits_processed = df.pfs_visit_id.to_list()
    visits_to_process = [v for v in visits if v not in visits_processed[1:]]
    return visits_to_process
 

def get_condition(visit0, config, ax1, ax2):
    conf = utils.read_conf(config)       
    opdb = qaDB.QaDB(conf["db"]["opdb"])
    qadb = qaDB.QaDB(conf["db"]["qadb"])

    visits = check_opdb(opdb, visit0)
    visits_to_process = check_qadb(qadb, visit0, visits[1:])
    print("visits to process: ", visits_to_process)
    cond=Condition(conf=config)
    cond.getConditionAg(visits=visits_to_process, showPlot=False, xaxis='agc_exposure_id', cc='flags', saveFig=True, figName='test', dirName=figDir)
    if len(visits)>0:
        if cond.df_seeing is not None:
            ndata = len(cond.df_seeing)
        else:
            print(f"# of data points = 0")
            sys.exit()
    else:
        ndata = 0
    print(f'# of data points = {ndata}')

    opdb.close()
    qadb.close()

    #ax1.plot(cond.df_seeing['taken_at'], cond.df_seeing['fwhm'])
    #ax2.plot(cond.df_transparency['taken_at'], cond.df_transparency['transp'])

def run(visit0, figDir, config, interval, f, wait=True):

    fig = plt.figure(figsize=(8,5))
    ax1 = fig.add_subplot(211)
    ax1.set_xlabel('taken_at')
    ax1.set_ylabel('Seeing (arcsec.)')
    ax1.set_ylim(0.0, 2.0)

    ax2 = fig.add_subplot(212)
    ax2.set_xlabel('taken_at')
    ax2.set_ylabel('Transparency')
    ax2.set_ylim(0.0, 2.0)

    base_time = time.time()
    next_time = 0
    while True:
        t = threading.Thread(target=f, args=(visit0, config, ax1, ax2))
        t.start()
        if wait:
            t.join()
        next_time = ((base_time - time.time()) % interval) or interval

        time.sleep(next_time)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        "--workDir",
        type=str,
        default='',
        help="path to the work directory",
    )
    parser.add_argument(
        "--configDir",
        type=str,
        default='configs',
        help="directory for configuration toml file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default='config.run22.qa.toml',
        help="configuration toml file",
    )
    parser.add_argument(
        "--figDir",
        type=str,
        default='figures',
        help="directory for figures",
    )
    parser.add_argument(
        "--visit0",
        type=int,
        default=999999,
        help="visit ID to start"
    )
    args = parser.parse_args()

    figDir=os.path.join(args.workDir, args.figDir)
    config=os.path.join(args.workDir, args.configDir, args.config)
    visit0=args.visit0

    run(visit0=visit0, figDir=figDir, config=config, interval=30, f=get_condition, wait=True)

