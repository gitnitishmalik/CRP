"""Shared fixtures. The tests read the pipeline's outputs, so run `python run_pipeline.py` first."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import portfolio  # noqa: E402
from config import PROCESSED  # noqa: E402

if not (PROCESSED / "anomalies.csv").exists():
    pytest.exit("data/processed is empty: run `python run_pipeline.py` before the tests", returncode=1)


@pytest.fixture(scope="session")
def book():
    return portfolio.load_book()


@pytest.fixture(scope="session")
def metrics():
    return json.loads((ROOT / "reports" / "model_metrics.json").read_text())
