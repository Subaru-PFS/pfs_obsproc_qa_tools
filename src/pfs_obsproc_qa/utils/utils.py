#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import toml


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
