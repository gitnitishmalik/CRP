"""3. Model building: the saved metrics reproduce, and the validation is set up as documented."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import RANDOM_STATE, ROOT
from evaluation import psi, summarize

REPORTS = ROOT / "reports"


def test_saved_test_metrics_reproduce_from_the_predictions(metrics):
    predictions = pd.read_csv(REPORTS / "test_predictions.csv")
    for kind in ["logistic", "lightgbm"]:
        actual = summarize(predictions.target_default, predictions[f"pd_{kind}"])
        for name, value in actual.items():
            assert np.isclose(value, metrics["models"][kind]["test"][name]), (kind, name)


def test_hold_out_customers_never_appear_in_training(book):
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
                       .split(book, book.target_default, book.customer_id))
    assert not set(book.customer_id.iloc[train]) & set(book.customer_id.iloc[test])
    predictions = pd.read_csv(REPORTS / "test_predictions.csv")
    assert set(predictions.account_id) == set(book.account_id.iloc[test])


def test_champion_rule_and_realistic_performance(metrics):
    gain = metrics["models"]["lightgbm"]["test"]["auc"] - metrics["models"]["logistic"]["test"]["auc"]
    assert np.isclose(gain, metrics["lightgbm_auc_gain_on_test"])
    assert metrics["champion"] == ("lightgbm" if gain >= metrics["materiality_auc"] else "logistic")
    # this data's honest ceiling is about 0.70; far above it would mean leakage
    assert 0.6 < metrics["models"][metrics["champion"]]["test"]["auc"] < 0.75


def test_protected_attributes_and_unreliable_fields_stay_out(metrics):
    used = set(metrics["logistic_features"]["numeric"]) | set(metrics["logistic_features"]["categorical"])
    assert not used & {"gender", "marital_status", "age", "loan_purpose", "collateral_flag", "customer_since"}
    assert not any(f.startswith("coll_") for f in used)
    age = metrics["design_checks"]["age"]
    assert age["adverse_impact_ratio_test_with"]["21-25"] < age["adverse_impact_ratio_test_without"]["21-25"]


def test_psi_is_zero_for_identical_samples_and_grows_with_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(size=5000)
    assert psi(base, base) < 1e-9
    assert psi(base, base + 1) > 0.25
    assert psi(list("aabbc") * 100, list("aabbc") * 100, categorical=True) < 1e-9
