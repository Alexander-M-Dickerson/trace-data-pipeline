# -*- coding: utf-8 -*-

from create_daily_standard_trace import *
import logging, sys, gc
gc.collect()

from _trace_settings import get_config

cfg = get_config("144a")       # brings start_date="2002-07-01", data_type="144a"
all_data = CreateDailyStandardTRACE(**cfg)