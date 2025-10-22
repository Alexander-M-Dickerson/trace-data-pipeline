# -*- coding: utf-8 -*-

from create_daily_standard_trace import *
import logging, sys, gc
gc.collect()

from _trace_settings import get_config

cfg = get_config("standard")   # brings start_date="2024-10-01", data_type="standard"
all_data = CreateDailyStandardTRACE(**cfg)