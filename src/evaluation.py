"""Model performance and stability measures used by training, the notebook and the dashboard."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y, p):
    """Largest gap between the score distributions of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def summarize(y, p):
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p)
    return {
        "n": int(len(y)),
        "default_rate": float(y.mean()),
        "avg_pd": float(p.mean()),
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
    }


def decile_table(y, p, bins=10):
    """Accounts ranked by predicted PD into equal-sized groups; group 1 = riskiest."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = bins - pd.qcut(df.p.rank(method="first"), bins, labels=False)
    out = df.groupby("decile").agg(accounts=("y", "size"), defaults=("y", "sum"), default_rate=("y", "mean"),
                                   avg_pd=("p", "mean"), min_pd=("p", "min"), max_pd=("p", "max"))
    out["share_of_defaults"] = out.defaults / out.defaults.sum()
    out["cumulative_share_of_defaults"] = out.share_of_defaults.cumsum()
    out["lift"] = out.default_rate / df.y.mean()
    return out.reset_index()


def psi(expected, actual, bins=10, categorical=False):
    """Population Stability Index of `actual` against the `expected` (baseline) distribution.

    Numbers are binned on the baseline's deciles; categories are compared one by one.
    Rule of thumb: < 0.10 stable, 0.10-0.25 investigate, > 0.25 significant shift.
    """
    if categorical:
        expected, actual = pd.Series(expected).astype(str), pd.Series(actual).astype(str)
        values = sorted(set(expected) | set(actual))
        e = expected.value_counts(normalize=True).reindex(values, fill_value=0).to_numpy()
        a = actual.value_counts(normalize=True).reindex(values, fill_value=0).to_numpy()
        e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
        return float(np.sum((a - e) * np.log(a / e)))
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # near-constant feature (e.g. a flag): compare the share of each distinct value instead
        values = np.union1d(np.unique(expected), np.unique(actual))
        e = np.array([(expected == v).mean() for v in values])
        a = np.array([(actual == v).mean() for v in values])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value):
    if value < 0.10:
        return "Stable"
    if value < 0.25:
        return "Investigate"
    return "Act"
