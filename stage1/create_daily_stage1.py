# -*- coding: utf-8 -*-
"""
create_daily_stage1.py
======================
Deployment wrapper for Stage 1 TRACE processing pipeline.
This script configures and executes the main pipeline (stage1_pipeline.py)
with settings from _stage1_settings.py.

This stage:
1. Loads treasury yields (Liu-Wu or FRED)
2. Loads TRACE data from stage0 outputs
3. Loads FISD bond characteristics
4. Merges FISD data with TRACE
5. Computes bond analytics (duration, convexity, credit spreads, etc.)
6. Merges credit ratings (S&P and Moody's)
7. Merges external identifiers via OSBAP linker
8. Applies ultra-distressed bond filters
9. Applies final price and date filters
10. Generates comprehensive data quality reports 

Author: Alex Dickerson
Created: 2025-11-17
"""

import sys
from pathlib import Path
import importlib.util

def run_stage1(config: dict):
    """
    Run Stage 1 pipeline by configuring and executing stage1_pipeline.py

    Parameters
    ----------
    config : dict
        Configuration dictionary from _stage1_settings.py

    Returns
    -------
    module
        The stage1_pipeline module with all processed data
    """

    # Import stage1_pipeline as a module
    stage1_dir = Path(config["stage1_dir"])
    pipeline_script_path = stage1_dir / "stage1_pipeline.py"

    if not pipeline_script_path.exists():
        raise FileNotFoundError(
            f"Pipeline script not found: {pipeline_script_path}\n"
            "Expected stage1_pipeline.py in stage1/ directory"
        )

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location("stage1_pipeline", pipeline_script_path)
    pipeline_module = importlib.util.module_from_spec(spec)
    sys.modules["stage1_pipeline"] = pipeline_module

    # Configure the module's global variables from our settings
    # This overwrites any defaults in stage1_pipeline.py

    # WRDS Configuration
    pipeline_module.WRDS_USERNAME = config["wrds_username"]
    pipeline_module.AUTHOR = config["author"]

    # Paths
    pipeline_module.ROOT_PATH = config["root_path"]
    pipeline_module.STAGE0_DIR = config["stage0_dir"]
    pipeline_module.STAGE1_DIR = config["stage1_dir"]
    pipeline_module.STAGE1_DATA = config["stage1_data"]
    pipeline_module.LOG_DIR = config["log_dir"]

    # Stage0 Input Files
    pipeline_module.STAGE0_DATE_STAMP = config["stage0_date_stamp"]
    pipeline_module.TRACE_MEMBERS = config["trace_members"]

    # Execution Settings
    pipeline_module.N_CORES = config["n_cores"]
    pipeline_module.N_CHUNKS = config["n_chunks"]

    # Date Filter
    pipeline_module.DATE_CUT_OFF = config["date_cut_off"]

    # Ultra Distressed Filter Configuration
    pipeline_module.ULTRA_DISTRESSED_CONFIG = config["ultra_distressed_config"]

    # Final Filters Configuration
    pipeline_module.FINAL_FILTER_CONFIG = config["final_filter_config"]

    # Yield Data Configuration
    pipeline_module.yld_type = config["yld_type"]
    pipeline_module.liu_wu_url = config["liu_wu_url"]

    # External Data URLs
    pipeline_module.LINKER_URL = config["linker_url"]
    pipeline_module.LINKER_ZIPKEY = config["linker_zipkey"]

    # Create directories (in case they don't exist yet)
    for d in [config["stage1_dir"], config["stage1_data"], config["log_dir"]]:
        d.mkdir(parents=True, exist_ok=True)

    # Now execute the module (this runs the configuration and import sections)
    spec.loader.exec_module(pipeline_module)

    # Call the main execution function
    print("\n" + "=" * 80)
    print("Starting Stage 1 Pipeline (via stage1_pipeline.py)")
    print("=" * 80)

    pipeline_module.run_all_steps()

    # Return the module so caller can access results if needed
    return pipeline_module


# For backward compatibility - allow direct import
if __name__ == "__main__":
    from _stage1_settings import get_config, validate_config

    config = get_config()
    validate_config(config)
    run_stage1(config)
