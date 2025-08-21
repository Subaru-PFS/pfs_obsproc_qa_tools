#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import toml
import enum


def read_conf(conf):
    """
    Parameters
    ----------
        conf: `str` config toml filepath (default: config.toml)
    
    Returns
    ----------
        config: return of toml.load
    
    Examples
    ----------

    """
    config = toml.load(conf)
    return config

def get_url(conf):
    url = f'{conf["dialect"]}://{conf["user"]}@{conf["host"]}:{conf["port"]}/{conf["dbname"]}'
    return url 

class OnsiteStatusType(enum.IntEnum):
    """Enumerated options for the status of the onsite processing"""
    INPROGRESS = 0
    COMPLETED = 1
    FAILED = 2
    def description(self):
        descriptions = {
            0: "The process is in progress",
            1: "The process was completed successfully", 
            2: "The process failed"
        }
        return descriptions[self.value]
