#!/usr/bin/env python

from .utils import get_url
import numpy as np
from sqlalchemy import create_engine, exc, text
import pandas as pd
from logzero import logger

import warnings
warnings.filterwarnings('ignore')


class QaDB(object):
    """QaDB

    Parameters
    ----------
    dbConfig : DB config read from toml file


    Examples
    ----------

    """

    def __init__(self, conf):
        self.conf = conf
        self._engine = create_engine(get_url(self.conf), future=True)
        self._conn = self._engine.raw_connection()

    def query(self, sqlCmd):
        df = pd.read_sql(sql=sqlCmd, con=self._conn)
        df = df.fillna(np.nan)
        return df

    def populateQATable(self, tableName, df, updateDB=True):
        """Populates a QA table with data from a DataFrame.

        Parameters:
            tableName (str): The name of the table to populate.
            df (pandas.DataFrame): The DataFrame containing the data to populate the table with.
            updateDB (bool): If True, existing records will not be updated. Default is True.

        Returns:
            None

        Raises:
            None

        Notes:
            - This is valid if the primary key is pfs_visit_id
            - If a record already exists, it will be updated with new information.
            - FIXME (this is not a smart way...) 
        """
        for idx, data in df.iterrows():
            df_new = pd.DataFrame(
                data={k: [v] for k, v in data.items()}
            )
            try:
                df_new = df_new.fillna(-1).astype(float)
            except:
                df_new = df_new
            try:
                if updateDB == True:
                    df_new.to_sql(tableName, self._engine,
                                  if_exists='append', index=False)
                else:
                    logger.info('No update...')
                    pass
            except exc.IntegrityError:
                if updateDB == True:
                    if 'pfs_visit_id' in data.keys():
                        pfs_visit_id = int(data['pfs_visit_id'])
                        keys = ''
                        vals = ''
                        for k, v in data.items():
                            keys = keys + k + ','
                            if isinstance(v, str):
                                vals = vals + f"'{v}',"
                            elif np.isnan(v):
                                vals = vals + "NULL,"
                            else:
                                vals = vals + str(v) + ','
                        if len(data) > 1:
                            sqlCmd = text(
                                f'UPDATE {tableName} SET ({keys[:-1]}) = ({vals[:-1]}) WHERE pfs_visit_id={pfs_visit_id};')
                        else:
                            sqlCmd = text(
                                f'UPDATE {tableName} SET {keys[:-1]} = {vals[:-1]} WHERE pfs_visit_id={pfs_visit_id};')
                        with self._engine.connect() as conn:
                            conn.execute(sqlCmd)
                            conn.commit()
                        logger.info(f'pfs_visit_id={pfs_visit_id} updated!')
                else:
                    logger.info('No update...')
                    pass

    def populateQATable2(self, tableName, df, updateDB=True):
        """Populates a QA table with data from a DataFrame.

        Parameters:
            tableName (str): The name of the table to populate.
            df (pandas.DataFrame): The DataFrame containing the data to populate the table with.
            updateDB (bool): If True, existing records will not be updated. Default is True.

        Returns:
            None

        Raises:
            None

        Notes:
            - This is valid if the primary key is a combination of pfs_visit_id and agc_exposure_id
            - If a record already exists, it will be updated with new information.
            - FIXME (this is not a smart way...) 
        """
        for idx, data in df.iterrows():
            df_new = pd.DataFrame(
                data={k: [v] for k, v in data.items()}
            )
            try:
                df_new = df_new.fillna(-1).astype(float)
            except:
                df_new = df_new
            try:
                if updateDB == True:
                    df_new.to_sql(tableName, self._engine,
                                if_exists='append', index=False)
                else:
                    logger.info('No update...')
                    pass

            except exc.IntegrityError:
                if updateDB == True:
                    if 'pfs_visit_id' in data.keys() and 'agc_exposure_id' in data.keys():
                        pfs_visit_id = int(data['pfs_visit_id'])
                        agc_exposure_id = int(data['agc_exposure_id'])
                        keys = ''
                        vals = ''
                        for k, v in data.items():
                            keys = keys + k + ','
                            if isinstance(v, str):
                                vals = vals + f"'{v}',"
                            elif np.isnan(v):
                                vals = vals + "NULL,"
                            else:
                                vals = vals + str(v) + ','
                        if len(data) > 1:
                            sqlCmd = text(
                                f'UPDATE {tableName} SET ({keys[:-1]}) = ({vals[:-1]}) WHERE pfs_visit_id={pfs_visit_id} AND agc_exposure_id={agc_exposure_id};')
                        else:
                            sqlCmd = text(
                                f'UPDATE {tableName} SET {keys[:-1]} = {vals[:-1]} WHERE pfs_visit_id={pfs_visit_id} AND agc_exposure_id={agc_exposure_id};')
                        with self._engine.connect() as conn:
                            conn.execute(sqlCmd)
                            conn.commit()
                        logger.info(
                            f'pfs_visit_id={pfs_visit_id} and agc_exposure_id={agc_exposure_id} updated!')
                else:
                    logger.info('No update...')
                    pass

    def close(self):
        self._conn.close()
