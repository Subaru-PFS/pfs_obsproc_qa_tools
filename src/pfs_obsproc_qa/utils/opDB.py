#!/usr/bin/env python

from sqlalchemy import create_engine
import pandas as pd
from logzero import logger

from .utils import get_url


class OpDB(object):
    """OpDB

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

    def close(self):
        self._conn.close()
        