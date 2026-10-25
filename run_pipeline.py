"""
Rebuilds every data product the dashboard needs, in order:

    1. src/ingest.py       clean and join the 5 raw tables     -> data/processed/accounts_clean.csv
    2. src/train_model.py  train, validate and score the model -> data/processed/scores.csv, models/, reports/
    3. src/anomaly.py      flag unusual accounts               -> data/processed/anomalies.csv

    python run_pipeline.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = ["ingest.py", "train_model.py", "anomaly.py"]

for step in STEPS:
    started = time.time()
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / "src" / step)], check=True, cwd=ROOT)
    print(f"--- {step} finished in {time.time() - started:.0f}s", flush=True)
