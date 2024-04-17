#!/usr/bin/env python

import numpy as np
from sqlalchemy import create_engine, exc, text
import pandas as pd
from logzero import logger

import warnings
warnings.filterwarnings('ignore')

from .utils import get_url


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
        self._engine = create_engine(get_url(self.conf))
        self._conn = self._engine.raw_connection()

    def query(self, sqlCmd):
        df = pd.read_sql(sql=sqlCmd, con=self._conn)
        return df

    def populateQATable(self, tableName, df):
        ''' FIXME (this is not a smart way...) '''
        for idx, data in df.iterrows():
            df_new = pd.DataFrame(
                data={k: [v] for k, v in data.items()}
                )
            try:
                df_new = df_new.fillna(-1).astype(float)
            except:
                df_new = df_new
            try:
                df_new.to_sql(tableName, self._engine, if_exists='append', index=False)
            except exc.IntegrityError:
                if 'pfs_visit_id' in data.keys():
                    pfs_visit_id = int(data['pfs_visit_id'])
                    keys = ''
                    vals = ''
                    for k, v in data.items():
                        keys = keys + k + ','
                        vals = vals + str(v) + ','
                    if len(data) > 1:
                        sqlCmd = text(f'UPDATE {tableName} SET ({keys[:-1]}) = ({vals[:-1]}) WHERE pfs_visit_id={pfs_visit_id};')
                    else:
                        sqlCmd = text(f'UPDATE {tableName} SET {keys[:-1]} = {vals[:-1]} WHERE pfs_visit_id={pfs_visit_id};')
                    with self._engine.connect() as conn:
                        conn.execute(sqlCmd)
                        conn.commit()
                    logger.info(f'pfs_visit_id={pfs_visit_id} updated!')
                else:
                    logger.info('No update...')
                    pass

    def close(self):
        self._conn.close()
        