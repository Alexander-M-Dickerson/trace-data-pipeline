# -*- coding: utf-8 -*-

from create_daily_standard_trace import *
import logging, sys, gc
gc.collect()

from _trace_settings import get_config

# The __main__ guard is REQUIRED now that stage0 may run its chunks in a process
# pool: on a spawn platform every child re-imports the main module, so without it
# each worker would restart the whole pipeline, recursively.
if __name__ == "__main__":
    cfg = get_config("standard")   # brings start_date="2024-10-01", data_type="standard"
    all_data = CreateDailyStandardTRACE(**cfg)
