"""Shared paths and constants for every step of the pipeline."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# The data ships without a snapshot date. The latest loan originates on 2023-12-30 and
# every loan carries 12 months of repayment history, so the last repayment month ends in
# December 2024. Using 2024-12-31 as the as-of date makes every repayment window complete.
AS_OF_DATE = pd.Timestamp("2024-12-31")

# Earliest age at which someone can have started working; caps impossible work histories.
MIN_WORKING_AGE = 16

RANDOM_STATE = 42

# Risk bands on the champion model's predicted probability of default: (upper bound, name).
RISK_BANDS = [(0.10, "Low"), (0.20, "Medium"), (0.35, "High"), (1.01, "Very High")]

# Illustrative loss-given-default by product, used only for the expected-loss view.
# The data has no recovery or collateral values (see docs/assumptions_log.md).
LGD = {"Mortgage": 0.20, "Auto Loan": 0.40, "Business Loan": 0.60, "Personal Loan": 0.65, "Credit Card": 0.75}
