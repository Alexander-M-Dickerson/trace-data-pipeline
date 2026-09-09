# -*- coding: utf-8 -*-

from create_daily_enhanced_trace import *
import logging, sys, gc
gc.collect()

from _trace_settings import get_config

# The __main__ guard is REQUIRED now that stage0 may run its chunks in a process
# pool. On a spawn platform (Windows) every child re-imports the main module, so
# without it each worker would start the whole pipeline again, recursively. WRDS is
# Linux and forks, where the guard costs nothing -- but it is what makes the same
# code safe to test anywhere.
if __name__ == "__main__":
    cfg = get_config("enhanced")
    all_data = CreateDailyEnhancedTRACE(**cfg)
