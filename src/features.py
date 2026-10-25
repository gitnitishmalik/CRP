"""
Model features, the two model recipes, and per-account explanations.

Shared by training (src/train_model.py), the notebooks and the dashboard so that a feature
means the same thing everywhere.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from config import RANDOM_STATE

CATEGORICAL = ["loan_type", "employment_type", "education_level", "region", "city_tier", "housing_type",
               "occupation_sector", "disbursal_channel"]

APPLICATION = ["annual_income", "years_employed", "num_dependents", "years_at_residence", "loan_amount",
               "interest_rate_pct", "tenure_months", "emi_to_income", "loan_to_income", "n_loans_customer"]

BUREAU = ["bureau_n_lines", "bureau_n_active", "bureau_n_written_off", "bureau_written_off_share",
          "bureau_n_dpd_30_plus", "bureau_max_dpd", "bureau_mean_dpd", "bureau_total_outstanding",
          "bureau_active_utilization", "bureau_debt_to_income", "bureau_share_non_bank", "bureau_no_file"]

REPAYMENT = ["rep_n_late", "rep_n_missed", "rep_max_dpd", "rep_mean_dpd", "rep_mean_dpd_last3",
             "rep_n_missed_last3", "rep_paid_ratio", "rep_max_missed_streak", "rep_months_since_missed",
             "rep_dpd_trend"]

NUMERIC = APPLICATION + BUREAU + REPAYMENT
FEATURES = CATEGORICAL + NUMERIC

# The logistic scorecard uses one feature per idea, each with the direction risk should move as it
# grows. Near-duplicates are left out (e.g. share of instalments paid mirrors missed payments
# exactly, bureau debt-to-income moves with total bureau debt), because duplicates split their weight
# and can flip signs, which would make the explanations misleading. Training then drops any feature
# whose fitted sign still disagrees with this table (see sign_checked_features in train_model.py).
EXPECTED_SIGN = {
    "annual_income": -1, "years_employed": -1, "years_at_residence": -1, "num_dependents": 1,
    "loan_to_income": 1, "interest_rate_pct": 1, "n_loans_customer": 1,
    "bureau_n_active": -1, "bureau_n_written_off": 1, "bureau_mean_dpd": 1, "bureau_n_dpd_30_plus": 1,
    "bureau_debt_to_income": 1, "bureau_active_utilization": 1, "bureau_share_non_bank": 1, "bureau_no_file": 1,
    "rep_n_late": 1, "rep_n_missed": 1, "rep_max_dpd": 1, "rep_n_missed_last3": 1, "rep_max_missed_streak": 1,
    "rep_months_since_missed": -1,
}

# right-skewed amounts and ratios get a log transform in the linear model
SKEWED = ["annual_income", "loan_amount", "emi_to_income", "loan_to_income", "bureau_total_outstanding",
          "bureau_debt_to_income"]

# Columns deliberately kept out of both models, with the reason (repeated in docs/model_report.md).
EXCLUDED = {
    "gender": "protected characteristic under ECOA / Regulation B",
    "marital_status": "protected characteristic under ECOA / Regulation B",
    "age": "protected characteristic under ECOA; dropping it cost no accuracy and made the model fairer to "
           "younger borrowers (docs/model_report.md)",
    "loan_purpose": "unrelated to the product it is attached to (data_quality_notes.md)",
    "collateral_flag": "repeats loan_type exactly",
    "customer_since": "dates contradict the customer's own loans",
    "collections (coll_*)": "unreliable dates and recorded after origination",
}

# Direction the risk is allowed to move as each feature grows (+1 up, -1 down) in LightGBM.
# Keeps the model's behaviour explainable: e.g. more missed payments can never lower the risk.
MONOTONE = {
    "annual_income": -1, "emi_to_income": 1, "loan_to_income": 1,
    "bureau_n_written_off": 1, "bureau_n_dpd_30_plus": 1, "bureau_max_dpd": 1, "bureau_mean_dpd": 1,
    "bureau_debt_to_income": 1,
    "rep_n_late": 1, "rep_n_missed": 1, "rep_max_dpd": 1, "rep_mean_dpd": 1, "rep_mean_dpd_last3": 1,
    "rep_n_missed_last3": 1, "rep_paid_ratio": -1, "rep_max_missed_streak": 1, "rep_months_since_missed": -1,
}

LABELS = {
    "loan_type": "Loan type", "employment_type": "Employment type", "education_level": "Education",
    "region": "Region", "city_tier": "City tier", "housing_type": "Housing", "occupation_sector": "Occupation sector",
    "disbursal_channel": "Disbursal channel",
    "age": "Age", "annual_income": "Annual income", "years_employed": "Years in current job",
    "num_dependents": "Dependents", "years_at_residence": "Years at current address", "loan_amount": "Loan amount",
    "interest_rate_pct": "Interest rate", "tenure_months": "Tenure (months)",
    "emi_to_income": "Instalment / monthly income", "loan_to_income": "Loan / annual income",
    "n_loans_customer": "Loans held with the bank",
    "bureau_n_lines": "Bureau: credit lines", "bureau_n_active": "Bureau: active lines",
    "bureau_n_written_off": "Bureau: written-off lines", "bureau_written_off_share": "Bureau: share of lines written off",
    "bureau_n_dpd_30_plus": "Bureau: lines 30+ days late", "bureau_max_dpd": "Bureau: worst days past due",
    "bureau_mean_dpd": "Bureau: average days past due", "bureau_total_outstanding": "Bureau: total owed elsewhere",
    "bureau_active_utilization": "Bureau: utilisation of active lines", "bureau_debt_to_income": "Bureau: debt / income",
    "bureau_share_non_bank": "Bureau: share from non-bank lenders", "bureau_no_file": "No bureau record",
    "rep_n_late": "Late payments (12 months)", "rep_n_missed": "Missed payments (12 months)",
    "rep_max_dpd": "Worst days past due (12 months)", "rep_mean_dpd": "Average days past due (12 months)",
    "rep_mean_dpd_last3": "Average days past due, last 3 months", "rep_n_missed_last3": "Missed payments, last 3 months",
    "rep_paid_ratio": "Share of instalments paid", "rep_max_missed_streak": "Longest run of missed payments",
    "rep_months_since_missed": "Months since last missed payment",
    "rep_dpd_trend": "Days-past-due trend (last vs first 3 months)",
}


class Winsorizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Clips each column to the percentiles seen in training so outliers can't dominate a linear model."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.low_ = np.nanquantile(X, self.lower, axis=0)
        self.high_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)


def category_levels(df):
    return {col: sorted(df[col].dropna().astype(str).unique()) for col in CATEGORICAL}


def model_frame(df, categories):
    """Feature matrix with categorical columns typed consistently for both models."""
    X = df[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = pd.Categorical(X[col].astype(str), categories=categories[col])
    return X


def make_logistic(C=0.3, numeric=None):
    numeric = list(EXPECTED_SIGN) if numeric is None else numeric
    skewed = make_pipeline(Winsorizer(), FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                           SimpleImputer(strategy="median"), StandardScaler())
    plain = make_pipeline(Winsorizer(), SimpleImputer(strategy="median"), StandardScaler())
    prep = ColumnTransformer([
        ("log", skewed, [f for f in numeric if f in SKEWED]),
        ("num", plain, [f for f in numeric if f not in SKEWED]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("model", LogisticRegression(C=C, max_iter=5000))])


def logistic_term(feature):
    """Name of a numeric feature's column after the logistic preprocessing."""
    return ("log__" if feature in SKEWED else "num__") + feature


def make_lightgbm(**params):
    settings = dict(n_estimators=500, learning_rate=0.02, num_leaves=15, min_child_samples=80, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, random_state=RANDOM_STATE,
                    n_jobs=8, verbose=-1, monotone_constraints=[MONOTONE.get(f, 0) for f in FEATURES])
    settings.update(params)
    return LGBMClassifier(**settings)


MONEY = {"annual_income", "loan_amount", "bureau_total_outstanding"}
PERCENT = {"emi_to_income", "bureau_active_utilization", "bureau_written_off_share", "bureau_share_non_bank",
           "rep_paid_ratio"}
MULTIPLE = {"loan_to_income", "bureau_debt_to_income"}


def format_value(feature, value):
    """Human-readable feature value for explanations."""
    if feature in CATEGORICAL:
        return str(value)
    if pd.isna(value):
        return "n/a"
    if feature in MONEY:
        return f"${value:,.0f}"
    if feature in PERCENT:
        return f"{value:.0%}"
    if feature in MULTIPLE:
        return f"{value:.2f}x"
    if feature == "interest_rate_pct":
        return f"{value:.1f}%"
    if feature == "bureau_no_file":
        return "yes" if value else "no"
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"


def _source_feature(transformed_name):
    """Map a transformed column name (e.g. 'cat__loan_type_Mortgage') back to its feature."""
    name = transformed_name.split("__", 1)[1]
    if transformed_name.startswith("cat__"):
        return max((f for f in CATEGORICAL if name.startswith(f + "_")), key=len)
    return name


def contributions(bundle, X):
    """Per-feature contribution to each account's log-odds of default, relative to an average account.

    Logistic regression: coefficient x (value - training mean), which is exactly its SHAP value.
    LightGBM: TreeSHAP values from LightGBM itself.
    """
    model, features = bundle["model"], bundle["features"]
    if bundle["kind"] == "logistic":
        prep, lr = model.named_steps["prep"], model.named_steps["model"]
        values = (prep.transform(X) - bundle["background_mean"]) * lr.coef_[0]
        sources = [_source_feature(n) for n in prep.get_feature_names_out()]
        grouping = pd.get_dummies(pd.Series(sources)).reindex(columns=features, fill_value=0).to_numpy(dtype=float)
        return pd.DataFrame(values @ grouping, columns=features, index=X.index)
    raw = model.predict(X[features], pred_contrib=True)
    return pd.DataFrame(raw[:, :-1], columns=features, index=X.index)
