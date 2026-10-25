"""1. Data ingestion and 2. analytics: the clean dataset and the shared portfolio numbers."""
import json

import numpy as np
import pandas as pd

import portfolio as pf
from config import PROCESSED, RAW


def test_one_row_per_loan_and_nothing_dropped(book):
    raw_loans = pd.read_csv(RAW / "loan_accounts.csv")
    assert len(book) == len(raw_loans) == 14_685
    assert book.account_id.is_unique
    assert book.customer_id.nunique() == 10_000


def test_every_key_check_passes_and_issues_are_logged():
    dq = pd.read_csv(PROCESSED / "dq_log.csv")
    keys = dq[dq.issue.str.contains("duplicate|missing from|recorded twice")]
    assert len(keys) == 9 and keys.rows_affected.eq(0).all()
    assert len(dq) == 45 and (dq.action != "check passed").sum() == 30


def test_missing_values_are_filled_and_flagged(book):
    for column in ["interest_rate_pct", "years_employed", "num_dependents", "years_at_residence"]:
        assert book[column].notna().all(), column
    assert book.interest_rate_derived.sum() > 0
    assert set(book.gender.unique()) == {"Male", "Female"}


def test_scores_and_explanations_for_every_account(book):
    assert book.pd.between(0, 1).all()
    assert book.drivers_json.map(lambda s: len(json.loads(s)) > 0).all()
    assert np.allclose(book.expected_loss, book.pd * book.lgd_assumed * book.exposure)
    # "riskier than x% of accounts": never 100%, because an account can't be riskier than itself
    assert book.pd_percentile.between(0, 1, inclusive="left").all()


def test_every_segment_dimension_reconciles_to_the_book(book):
    for dim in pf.DIMENSIONS:
        table = pf.segment_summary(book, dim)
        assert table.accounts.sum() == len(book), dim
        assert np.isclose(table.exposure.sum(), book.exposure.sum()), dim


def test_filters_and_cross_tab(book):
    south_mortgages = pf.apply_filters(book, {"region": ["South"], "loan_type": ["Mortgage"]})
    expected = book[book.region.eq("South") & book.loan_type.eq("Mortgage")]
    assert south_mortgages.account_id.tolist() == expected.account_id.tolist()
    rate, count = pf.cross_tab(book, "income_band", "employment_type")
    assert count.to_numpy().sum() == len(book)
    assert list(rate.index) == pf.ORDER["income_band"]


def test_the_five_insights(book):
    base, exposure = book.target_default.mean(), book.exposure.sum()

    def insight(mask):
        return round(book.exposure[mask].sum() / exposure, 3), round(book.target_default[mask].mean() / base - 1, 2)

    assert insight(book.loan_type.eq("Mortgage")) == (0.523, 0.53)
    assert insight(book.emi_to_income.ge(0.3)) == (0.326, 0.59)
    assert insight(book.income_band.isin(["<30k", "30-50k"])) == (0.127, 0.74)
    assert insight(book.employment_type.eq("Unemployed") & book.income_band.eq("<30k")) == (0.016, 1.75)
    assert insight(book.region.eq("South") & book.loan_type.eq("Mortgage")) == (0.140, 0.70)
