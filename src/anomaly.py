"""
Bonus AI feature (beyond the brief's one): anomaly detection.

Flags accounts whose repayment, bureau or affordability pattern is unusual compared with their
peer segment (same loan type and income band), and explains each flag in plain words.

    python src/anomaly.py        (after src/train_model.py)

Triggers (an account is flagged if any applies)
    Peer outlier          Isolation Forest on the account's position within its peer group
                          (percentile of each risk metric, risky direction only); the 5% most
                          unusual are flagged
    Affordability breach  monthly instalment is larger than the borrower's monthly income
    Payment shock         2+ missed payments in the last 3 months after 9 months with no missed payment
                          (vs own history)

Severity is High for affordability breaches and the 1% most unusual peer outliers, Medium otherwise.
Flags go to an analyst review queue in the dashboard; nothing happens to a customer automatically.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import PROCESSED, RANDOM_STATE, ROOT
from features import LABELS, format_value
from portfolio import load_book

# metrics where a higher value means more risk; the detector only looks at these
RISK_METRICS = ["rep_mean_dpd", "rep_n_missed", "rep_n_missed_last3", "rep_max_missed_streak", "rep_mean_dpd_last3",
                "bureau_mean_dpd", "bureau_n_written_off", "bureau_debt_to_income", "emi_to_income", "loan_to_income"]
CONTAMINATION = 0.05      # share of accounts flagged as peer outliers
HIGH_SEVERITY = 0.01      # most unusual share that is High severity
REASON_PERCENTILE = 0.90  # a metric is a reason when it is worse than this share of peers
MIN_PEER_SIZE = 50        # smaller peer groups fall back to the whole loan type


def peer_groups(book):
    peer = book.loan_type + " | " + book.income_band
    small = peer.map(peer.value_counts()) < MIN_PEER_SIZE
    return peer.where(~small, book.loan_type + " | all incomes")


def detect(book):
    metrics = book[RISK_METRICS].fillna(0)
    peer = peer_groups(book)
    # share of peers with a value at or below this account's value (1.0 = worst in the group)
    percentile = metrics.groupby(peer).rank(pct=True, method="max")
    # share of peers with a strictly lower value, used to word the reasons
    below = (metrics.groupby(peer).rank(method="min") - 1).div(metrics.groupby(peer).transform("size"), axis=0)
    peer_median = metrics.groupby(peer).transform("median")

    # only the risky direction counts: anything at or below the peer median is treated as normal,
    # so the forest can't flag accounts for being unusually *good*
    adverse = (percentile - 0.5).clip(lower=0)
    forest = IsolationForest(n_estimators=400, random_state=RANDOM_STATE).fit(adverse)
    unusualness = pd.Series(-forest.score_samples(adverse), index=book.index)
    score = unusualness.rank(pct=True)

    peer_outlier = score > 1 - CONTAMINATION
    affordability = book.emi_exceeds_income.eq(1)
    shock = book.rep_n_missed_last3.ge(2) & book.rep_n_missed.eq(book.rep_n_missed_last3)

    reasons = []
    for i in book.index:
        found = []
        if affordability[i]:
            found.append(f"Instalment is {book.emi_to_income[i]:.0%} of monthly income (affordability breach)")
        if shock[i]:
            found.append(f"{book.rep_n_missed_last3[i]:.0f} missed payments in the last 3 months after 9 months "
                         "with no missed payment")
        if peer_outlier[i]:
            row = below.loc[i].sort_values(ascending=False)
            worst = row[row >= REASON_PERCENTILE].index[:3]
            prefix = ""
            if len(worst) == 0:
                # nothing extreme on its own: the forest reacted to several moderately high metrics together
                worst = row[row >= 0.5].index[:3]
                prefix = "Unusual combination for its peer group: "
            parts = [f"{LABELS[m]} {format_value(m, metrics.at[i, m])} vs peer median "
                     f"{format_value(m, peer_median.at[i, m])} (higher than {int(row[m] * 100)}% of peers)"
                     for m in worst]
            if prefix:
                found.append(prefix + ", ".join(parts))
            else:
                found.extend(parts)
        reasons.append("; ".join(found))

    triggers = (pd.DataFrame({"Affordability breach": affordability, "Payment shock": shock,
                              "Peer outlier": peer_outlier})
                .apply(lambda r: ", ".join(name for name, hit in r.items() if hit), axis=1))
    flag = affordability | shock | peer_outlier
    severity = np.select([affordability | (score > 1 - HIGH_SEVERITY), flag], ["High", "Medium"], default="")
    return pd.DataFrame({
        "account_id": book.account_id,
        "peer_group": peer,
        "anomaly_score": score.round(4),
        "anomaly_flag": flag.astype(int),
        "anomaly_severity": severity,
        "anomaly_trigger": triggers,
        "anomaly_reasons": reasons,
    })


def evaluate(book, flags):
    """How much the flags concentrate risk. The target is used only here, never to build the flags."""
    flagged = flags.anomaly_flag.eq(1)
    y = book.target_default
    base = y.mean()
    model_high = book.risk_band.isin(["High", "Very High"])
    by_trigger = {}
    for trigger in ["Peer outlier", "Affordability breach", "Payment shock"]:
        hit = flags.anomaly_trigger.str.contains(trigger)
        by_trigger[trigger] = {"accounts": int(hit.sum()), "default_rate": float(y[hit].mean()),
                               "lift": float(y[hit].mean() / base)}
    blind = flagged & ~model_high
    return {
        "accounts": int(len(book)),
        "flagged": int(flagged.sum()),
        "flagged_share": float(flagged.mean()),
        "portfolio_default_rate": float(base),
        "flagged_default_rate": float(y[flagged].mean()),
        "unflagged_default_rate": float(y[~flagged].mean()),
        "lift": float(y[flagged].mean() / base),
        "precision": float(y[flagged].mean()),
        "recall": float(flagged[y.eq(1)].mean()),
        "share_flagged_also_model_high_risk": float(model_high[flagged].mean()),
        "flagged_but_model_low_or_medium": {"accounts": int(blind.sum()), "default_rate": float(y[blind].mean())},
        "by_trigger": by_trigger,
        "by_severity": {s: {"accounts": int(flags.anomaly_severity.eq(s).sum()),
                            "default_rate": float(y[flags.anomaly_severity.eq(s)].mean())} for s in ["High", "Medium"]},
    }


def main():
    book = load_book()
    if "anomaly_flag" in book:
        book = book.drop(columns=[c for c in book.columns if c.startswith("anomaly_") or c == "peer_group"])
    flags = detect(book)
    flags.to_csv(PROCESSED / "anomalies.csv", index=False)
    report = evaluate(book, flags)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "anomaly_eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
