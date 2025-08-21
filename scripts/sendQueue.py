#!/usr/bin/env python3

import os
import argparse
import subprocess
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import time
import toml
from datetime import datetime, timedelta

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

def retrieve_data_from_opdb(config, visitStart, visitEnd):
    # Connect to the database
    engine = create_engine(config['database']['opdb']['url'])
    minExptime = config['general']['min_exptime']

    # Query the "sps_visit" and "sps_exposure" table
    query = f"SELECT sps_visit.pfs_visit_id,sps_exposure.time_exp_end FROM sps_visit JOIN sps_exposure ON sps_visit.pfs_visit_id=sps_exposure.pfs_visit_id WHERE sps_visit.exp_type='object' AND sps_exposure.exptime > {minExptime} AND sps_visit.pfs_visit_id BETWEEN {visitStart} AND {visitEnd} ORDER BY sps_visit.pfs_visit_id DESC;"

    df = pd.read_sql(query, engine)

    # Close the database connection
    engine.dispose()

    return df

def retrieve_num_targets_from_opdb(config, visit):
    # Connect to the database
    engine = create_engine(config['database']['opdb']['url'])

    # Query the "sps_visit" and "sps_exposure" table

    query = f"SELECT pfs_design.num_sci_designed,pfs_design.num_cal_designed,pfs_design.num_sky_designed FROM pfs_visit JOIN pfs_design ON pfs_visit.pfs_design_id=pfs_design.pfs_design_id WHERE pfs_visit.pfs_visit_id={visit};"

    df = pd.read_sql(query, engine)

    # Close the database connection
    engine.dispose()

    return df

def retrieve_data_from_registry(config, visitStart, visitEnd):
    minExptime = config['general']['min_exptime']

    # Connect to the database
    engine = create_engine(config['database']['registry']['url'])

    # Query the "raw" table
    query = f'SELECT * FROM visit WHERE id BETWEEN {visitStart} AND {visitEnd} AND exposure_time > {minExptime};'
    df = pd.read_sql(query, engine)

    # Close the database connection
    engine.dispose()

    return df

def check_raw_data_exist(config, visit, time_exp_end):
    obstime = datetime.utcfromtimestamp(time_exp_end.astype(int) * 1e-9) + timedelta(hours=10)
    obsdate = obstime.date().strftime('%Y%m%d')
    dirBaseName = f"/work/datastore/PFS/raw/sps/raw/{obsdate}/{visit}"
    nRaws = len([file for file in os.listdir(dirBaseName) if file.startswith('raw_PFS') and file.endswith('.fits')])
    return nRaws

def generate_files(visit, 
                   workdir='.', 
                   template_dir='templates', 
                   output_dir='.',
                   input='PFS/default',
                   output='test',
                   version='w.2025.10',
                   config_qa="config.s25a.qa.toml"
                   ):
    """
    Generate files for a given visit.

    Args:
        visit (int): The visit number.
        workdir (str, optional): The working directory. Defaults to '.'.
        template_dir (str, optional): The directory containing the template files. Defaults to 'templates'.
        output_dir (str, optional): The directory to save the generated files. Defaults to '.'.
        input (str, optional): The input collections. Defaults to 'PFS/default'.
        output (str, optional): The output collections. Defaults to 'test'.
    """
    # Set the paths of the template files
    sh_template_path = os.path.join(template_dir, 'process_visits.sh')
    pbs_template_path = os.path.join(template_dir, 'process_visits.pbs')

    # Read the template files
    with open(sh_template_path, 'r') as f:
        sh_template = f.read()
    with open(pbs_template_path, 'r') as f:
        pbs_template = f.read()

    # Generate new files using the visit number
    sh_content = sh_template.replace('WORKDIR_TEMPLATE', workdir)
    sh_content = sh_content.replace('VISITS_TEMPLATE', str(visit))
    sh_content = sh_content.replace('INPUT_TEMPLATE', input)
    sh_content = sh_content.replace('OUTPUT_TEMPLATE', output)
    sh_content = sh_content.replace('VERSION_TEMPLATE', version)
    sh_content = sh_content.replace('CONFIG_TEMPLATE', config_qa)
    pbs_content = pbs_template.replace('WORKDIR_TEMPLATE', workdir)
    pbs_content = pbs_content.replace('VISITS_TEMPLATE', str(visit))
    pbs_content = pbs_content.replace('LOGFILE_TEMPLATE', f'v{visit}')

    # Save the new files
    with open(os.path.join(output_dir, f'process_{visit}.sh'), 'w') as f:
        f.write(sh_content)
    with open(os.path.join(output_dir, f'process_{visit}.pbs'), 'w') as f:
        f.write(pbs_content)
    print(f"files for v{visit} generated")

def submit_pbs(pbs_file):
    subprocess.run(['qsub', pbs_file])
    print(f"{pbs_file} sent")

def update_status(config_qa, visit, status=0):
    conf = toml.load(config_qa)
    qadb = qaDB.QaDB(conf["db"]["qadb"])
    qadb.set_onsite_processing_status(pfs_visit_id=visit, status=status)
    qadb.close()    
    return 0

def main(config, visitStart, visitEnd, visits):
    """
    Process the data and send a queue for each visit within the specified range.

    Args:
        config (dict): Configuration settings.
        visitStart (int): Starting visit ID.
        visitEnd (int): Ending visit ID.
        visits (list of int): a list of visits to process

    Returns:
        None
    """

    workdir = config['general']['workdir']
    output_dir = config['general']['output_dir']
    template_dir = config['general']['template_dir']
    time_interval = config['general']['interval']
    drp_input = config['drp']['input']
    drp_output = config['drp']['output']
    drp_version = config['drp']['version']
    ncam = config['general']['ncam']
    config_qa = config['qa']['config']

    visitsProcessed = []

    if visits is None:
        # Continue running the code
        while True:

            # Retrieve data from DRP registry
            df_regist = retrieve_data_from_registry(config, visitStart, visitEnd)
            visits_regist = df_regist['id'].astype('i4').to_numpy()
            visits_regist_unique = np.unique(visits_regist)
            
            # Retrieve data from opDB
            df_opdb = retrieve_data_from_opdb(config, visitStart, visitEnd)
            visits_opdb = df_opdb['pfs_visit_id'].astype('i4').to_numpy()
            visits_opdb_unique = np.unique(visits_opdb)
            time_exp_ends = {v: df_opdb.time_exp_end[visits_opdb == v].values[0] for v in visits_opdb_unique}

            for visit in visits_regist_unique:
                if visit not in visitsProcessed:

                    # get pfs_visit_id stored in registry
                    ncam_regist = len(visits_regist[visits_regist==visit])
                    ncam_opdb = len(visits_opdb[visits_opdb==visit])
                    #print(f"The number of records (opDB/registry): {ncam_opdb}/{ncam_regist}")

                    # get number of targets 
                    df = retrieve_num_targets_from_opdb(config, visit)
                    
                    if len(df)==1:
                        ncals = df['num_cal_designed'][0]
                    else:
                        ncals = 0

                    # check the elapsed time since the exposure is done
                    try:
                        dt = np.datetime64(datetime.now()) - time_exp_ends[visit]
                        dt /= 1e+09
                        dt = float(dt)
                    except Exception as e:
                        dt = 0

                    # reduce only visits with opDB and registry is consistent (remove?)
                    #if ncam_regist == ncam_opdb:

                    if visit in time_exp_ends.keys():
                        n_raws = check_raw_data_exist(config_qa, visit, time_exp_ends[visit])
                    else:
                        n_raws = 0

                    if ncam_opdb == ncam  and ncam_regist == 1 and n_raws == ncam:

                        # reduce only visits with the number of FLUXSTDs > 0
                        if ncals > 0:

                            generate_files(visit, 
                                        workdir=workdir, 
                                        template_dir=template_dir, 
                                        output_dir=output_dir,
                                        input=drp_input,
                                        output=drp_output,
                                        version=drp_version,
                                        config_qa=config_qa,
                                        )
                            if args.sendQueue:
                                submit_pbs(os.path.join(output_dir, f'process_{visit}.pbs'))
                            update_status(config_qa, visit, status=utils.OnsiteStatusType.INPROGRESS)
                            print(f"    The number of FLUXSTDs: {ncals}")
                            visitsProcessed.append(visit)
                        time.sleep(time_interval)
                    else:
                        if ncam_regist == 1 and dt > 900. and n_raws > 0:
                            # reduce only visits with the number of FLUXSTDs > 0
                            if ncals > 0:
                                generate_files(visit, 
                                            workdir=workdir, 
                                            template_dir=template_dir, 
                                            output_dir=output_dir,
                                            input=drp_input,
                                            output=drp_output,
                                            version=drp_version,
                                            config_qa=config_qa,
                                            )
                                if args.sendQueue:
                                    submit_pbs(os.path.join(output_dir, f'process_{visit}.pbs'))
                                update_status(config_qa, visit, status=utils.OnsiteStatusType.INPROGRESS)
                                print(f"    The number of FLUXSTDs: {ncals}")
                                visitsProcessed.append(visit)
                            time.sleep(time_interval)

            # wait for a while
            time.sleep(time_interval)
    else:
        for visit in visits:
            if visit not in visitsProcessed:
                # get number of targets 
                df = retrieve_num_targets_from_opdb(config, visit)
                
                if len(df)==1:
                    ncals = df['num_cal_designed'][0]
                else:
                    ncals = 0

                # reduce only visits with the number of FLUXSTDs > 0
                generate_files(visit, 
                            workdir=workdir, 
                            template_dir=template_dir, 
                            output_dir=output_dir,
                            input=drp_input,
                            output=drp_output,
                            version=drp_version,
                            config_qa=config_qa,
                            )
                if args.sendQueue:
                    submit_pbs(os.path.join(output_dir, f'process_{visit}.pbs'))
                update_status(config_qa, visit, status=utils.OnsiteStatusType.INPROGRESS)
                print(f"    The number of FLUXSTDs: {ncals}")
                visitsProcessed.append(visit)
                time.sleep(time_interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--config', type=str, default="configs/config.toml", help='The path to the config toml file')
    parser.add_argument('--visitStart', type=int, default=None, help='The minimum visit ID to process')
    parser.add_argument('--visitEnd', type=int, default=999999, help='The maximum visit ID to process')
    parser.add_argument('--visits', type=int, nargs='+', default=None, help='If this is specified, just process the specified visits instead of loop processing')
    parser.add_argument('--sendQueue', action='store_true', help='Send the created script to queue')
    args = parser.parse_args()

    config = toml.load(args.config)

    if (type(args.visitStart) is int and type(args.visitEnd) is int) or args.visits is not None:
        main(config=config, 
             visitStart=args.visitStart, 
             visitEnd=args.visitEnd,
             visits=args.visits)
    else:
        print(f"exit")

