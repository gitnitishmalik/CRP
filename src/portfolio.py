"""
Portfolio analytics shared by the analysis notebook, the dashboard and the chat tools,
so every number shown anywhere comes from the same code.

Definitions used everywhere
    default rate  share of loan accounts with target_default = 1
    exposure      original loan amount (reported balances don't reconcile, see data_quality_notes.md)
    lift          segment default rate / default rate of the population it is compared with
"""
import numpy as np
import pandas as pd

from config import PROCESSED

DIMENSIONS = {
    "loan_type": "Loan type",
    "income_band": "Income band",
    "employment_type": "Employment type",
    "region": "Region",
    "age_band": "Age band",
    "city_tier": "City tier",
    "education_level": "Education",
    "housing_type": "Housing",
    "occupation_sector": "Occupation sector",
    "disbursal_channel": "Disbursal channel",
    "emi_to_income_band": "Instalment / income",
    "lti_band": "Loan / income",
    "origination_year": "Origination year",
    "risk_band": "Model risk band",
}

ORDER = {
    "income_band": ["<30k", "30-50k", "50-75k", "75-100k", "100k+"],
    "age_band": ["21-25", "26-35", "36-45", "46-55", "56+"],
    "city_tier": ["Tier 1", "Tier 2", "Tier 3"],
    "emi_to_income_band": ["<10%", "10-20%", "20-30%", "30-50%", "50%+"],
    "lti_band": ["<0.25x", "0.25-0.5x", "0.5-1x", "1-2x", "2x+"],
    "risk_band": ["Low", "Medium", "High", "Very High"],
}

# segments smaller than this get a warning instead of a confident conclusion
SMALL_SEGMENT = 100


def load_book():
    """Clean accounts plus model scores and anomaly flags when those steps have run."""
    book = pd.read_csv(PROCESSED / "accounts_clean.csv",
                       parse_dates=["origination_date", "customer_since", "first_loan_date", "relationship_start"])
    for name in ["scores.csv", "anomalies.csv"]:
        path = PROCESSED / name
        if path.exists():
            book = book.merge(pd.read_csv(path), on="account_id", how="left", validate="1:1")
    text = [c for c in ["top_risk_drivers", "anomaly_severity", "anomaly_trigger", "anomaly_reasons"] if c in book]
    book[text] = book[text].fillna("")
    return book


def apply_filters(df, filters=None):
    """Keep rows whose columns match every {column: [allowed values]} entry; empty lists are ignored."""
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for column, values in filters.items():
        if values is None or (isinstance(values, (list, tuple, set)) and len(values) == 0):
            continue
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        mask &= df[column].astype(str).isin([str(v) for v in values])
    return df[mask]


def sort_segments(table, dim, column="segment"):
    """Put segments of ordinal dimensions (bands, tiers) in their natural order."""
    if dim in ORDER:
        rank = {value: i for i, value in enumerate(ORDER[dim])}
        return table.sort_values(column, key=lambda s: s.astype(str).map(rank)).reset_index(drop=True)
    return table.sort_values(column).reset_index(drop=True)


def overview(df):
    defaulted_exposure = df.exposure[df.target_default.eq(1)].sum()
    summary = {
        "accounts": int(len(df)),
        "customers": int(df.customer_id.nunique()),
        "exposure": float(df.exposure.sum()),
        "default_rate": float(df.target_default.mean()) if len(df) else float("nan"),
        "defaulted_accounts": int(df.target_default.sum()),
        "defaulted_exposure": float(defaulted_exposure),
        "defaulted_exposure_share": float(defaulted_exposure / df.exposure.sum()) if len(df) else float("nan"),
    }
    if "pd" in df:
        summary["avg_predicted_pd"] = float(df.pd.mean())
        summary["expected_loss"] = float(df.expected_loss.sum())
    if "anomaly_flag" in df:
        summary["flagged_anomalies"] = int(df.anomaly_flag.sum())
    return summary


def segment_summary(df, dim):
    """One row per segment: size, exposure, default rate, and lift against the whole of `df`."""
    work = df.assign(defaulted_exposure=df.exposure * df.target_default)
    g = work.groupby(dim, observed=True)
    out = pd.DataFrame({
        "accounts": g.size(),
        "exposure": g.exposure.sum(),
        "default_rate": g.target_default.mean(),
        "defaulted_exposure": g.defaulted_exposure.sum(),
    })
    out["account_share"] = out.accounts / out.accounts.sum()
    out["exposure_share"] = out.exposure / out.exposure.sum()
    out["share_of_defaulted_exposure"] = out.defaulted_exposure / out.defaulted_exposure.sum()
    out["lift"] = out.default_rate / df.target_default.mean()
    if "pd" in df:
        out["avg_predicted_pd"] = g.pd.mean()
    if "anomaly_flag" in df:
        out["flagged_anomalies"] = g.anomaly_flag.sum()
    out["small_sample"] = out.accounts < SMALL_SEGMENT
    out = out.reset_index().rename(columns={dim: "segment"})
    out["segment"] = out.segment.astype(str)
    return sort_segments(out, dim)


def cross_tab(df, row, col):
    """Default rate and account counts for every row x col combination."""
    rate = pd.crosstab(df[row], df[col], values=df.target_default, aggfunc="mean")
    count = pd.crosstab(df[row], df[col])
    for dim, axis in [(row, 0), (col, 1)]:
        if dim in ORDER:
            order = [v for v in ORDER[dim] if v in (rate.index if axis == 0 else rate.columns)]
            rate = rate.reindex(order, axis=axis)
            count = count.reindex(order, axis=axis)
    return rate, count


def vintage_trend(df, by="origination_year"):
    work = df.assign(defaulted_exposure=df.exposure * df.target_default)
    g = work.groupby(by)
    out = pd.DataFrame({
        "accounts": g.size(),
        "exposure": g.exposure.sum(),
        "default_rate": g.target_default.mean(),
    })
    if "pd" in df:
        out["avg_predicted_pd"] = g.pd.mean()
    return out.reset_index()


def wilson_interval(defaults, n, z=1.96):
    """95% confidence interval for a default rate; wide intervals mean small segments."""
    if n == 0:
        return (np.nan, np.nan)
    p = defaults / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half
