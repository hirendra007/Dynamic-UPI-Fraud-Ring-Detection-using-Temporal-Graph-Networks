"""
run_pipeline.py - Master Orchestrator for Real-Time Fraud & Graph Analytics Pipeline
Root launcher.
"""

import sys
from pathlib import Path

# Redirect to fraud-pipeline/run_pipeline.py if available
fp_runner = Path(__file__).resolve().parent / "fraud-pipeline" / "run_pipeline.py"
if fp_runner.exists():
    import runpy
    sys.path.insert(0, str(fp_runner.parent))
    runpy.run_path(str(fp_runner), run_name="__main__")
