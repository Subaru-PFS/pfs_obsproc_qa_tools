#!/usr/bin/env python

import numpy as np
import pandas as pd
import toml

SEEING_NOMINAL=0.80        # arcsec
TRANSPARENCY_NOMINAL=0.90  # 
NOISE_LEVEL_NOMINAL=40.0   # ADU?

class ExposureTime(object):
    """Observing condition

    Parameters
    ----------
    visits : `pfs_visit_id` or `list` of `pfs_visit_id`


    Examples
    ----------

    """
    def __init__(self):
        self.SEEING_NOMINAL = SEEING_NOMINAL
        self.TRANSPARENCY_NOMINAL = TRANSPARENCY_NOMINAL
        self.NOISE_LEVEL_NOMINAL = NOISE_LEVEL_NOMINAL

    def calcEffectiveExposureTime(self):
        return 0

