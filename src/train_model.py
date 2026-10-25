"""
Step 3 - Model building.

Trains a logistic regression baseline and a LightGBM challenger to predict target_default,
validates them three ways, picks a champion, and scores every account with its key drivers.

    python src/train_model.py

Validation
    1. Hold-out test set: 20% of customers. All loans of a customer sit on one side of the split,
       so their shared bureau data can't leak from training into testing.
    2. 5-fold cross-validation grouped by customer, on the training part, used for tuning.
    3. Out-of-time: train on loans originated 2018-2021, test on 2022-2023 loans of customers
       the model has never seen.

Design checks, saved with the metrics so the documents quote numbers the code produces: the
hold-out accuracy and age-band fairness with age put back in, and the accuracy with collections
features added. Both stay out of the model.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from config import LGD, PROCESSED, RANDOM_STATE, RISK_BANDS, ROOT
from evaluation import decile_table, psi, psi_status, summarize
from features import (CATEGORICAL, EXCLUDED, EXPECTED_SIGN, FEATURES, LABELS, category_levels, contributions,
                      format_value, logistic_term, make_lightgbm, make_logistic, model_frame)

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

LOGISTIC_GRID = [{"C": c} for c in [0.03, 0.1, 0.3, 1.0]]
LIGHTGBM_GRID = [
    {"num_leaves": 7, "min_child_samples": 100, "n_estimators": 700},
    {"num_leaves": 15, "min_child_samples": 80, "n_estimators": 500},
    {"num_leaves": 31, "min_child_samples": 60, "n_estimators": 300},
]

# LightGBM must beat the simpler, fully explainable model by this much test AUC to become champion
MATERIALITY = 0.01

# collections columns re-tested as model inputs in design_checks (they stay out of the model)
COLLECTIONS = ["coll_n_events", "coll_n_events_raw", "coll_total_recovered"]


def build(kind, params):
    return make_logistic(**params) if kind == "logistic" else make_lightgbm(**params)


def sign_checked_features(X, y):
    """Scorecard sign check: drop, one at a time, the weakest feature whose coefficient points the
    opposite way to EXPECTED_SIGN (a symptom of overlap with another feature), until all agree."""
    numeric, dropped = list(EXPECTED_SIGN), []
    while True:
        model = make_logistic(numeric=numeric).fit(X, y)
        coef = pd.Series(model.named_steps["model"].coef_[0], index=model.named_steps["prep"].get_feature_names_out())
        wrong = [f for f in numeric if np.sign(coef[logistic_term(f)]) != EXPECTED_SIGN[f]]
        if not wrong:
            return numeric, dropped
        weakest = min(wrong, key=lambda f: abs(coef[logistic_term(f)]))
        print(f"  sign check: dropping {weakest} (coefficient {coef[logistic_term(weakest)]:+.3f})")
        numeric.remove(weakest)
        dropped.append(weakest)


def tune(kind, grid, X, y, groups):
    results = []
    for params in grid:
        aucs = []
        for tr, te in GroupKFold(5).split(X, y, groups):
            model = build(kind, params).fit(X.iloc[tr], y.iloc[tr])
            aucs.append(roc_auc_score(y.iloc[te], model.predict_proba(X.iloc[te])[:, 1]))
        results.append({"model": kind, "params": params, "cv_auc_mean": float(np.mean(aucs)),
                        "cv_auc_std": float(np.std(aucs))})
        print(f"  {kind:<9} {params}  CV AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
    return max(results, key=lambda r: r["cv_auc_mean"]), results


def risk_band(pd_values):
    edges = [0] + [upper for upper, _ in RISK_BANDS]
    return pd.cut(pd_values, edges, labels=[name for _, name in RISK_BANDS], right=False).astype(str)


def explain(book, contrib, top=8):
    """Top drivers per account as text (risk-raising only) and as JSON (both directions)."""
    features = list(contrib.columns)
    values = book[features].to_numpy(dtype=object)
    impacts = contrib.to_numpy()
    strongest = np.argsort(-np.abs(impacts), axis=1)[:, :top]
    texts, payloads = [], []
    for i in range(len(impacts)):
        payload = [{"feature": features[j], "label": LABELS[features[j]],
                    "value": format_value(features[j], values[i, j]), "impact": round(float(impacts[i, j]), 4)}
                   for j in strongest[i]]
        raising = [d for d in payload if d["impact"] > 0][:3]
        texts.append("; ".join(f"{d['label']}: {d['value']}" for d in raising))
        payloads.append(json.dumps(payload))
    return texts, payloads


def adverse_impact_ratio(share_high_risk):
    """Share of each group NOT rated High / Very High, relative to the most favoured group.
    The four-fifths rule asks for at least 0.8."""
    not_high = 1 - share_high_risk
    return not_high / not_high.max()


def fairness_table(book, scores):
    """Outcome and prediction by protected group. None of these attributes is a model input."""
    high = scores.risk_band.isin(["High", "Very High"])
    rows = []
    for attribute in ["gender", "marital_status", "age_band"]:
        g = pd.DataFrame({"group": book[attribute], "y": book.target_default, "pd": scores.pd, "high": high})
        t = g.groupby("group").agg(accounts=("y", "size"), actual_default_rate=("y", "mean"),
                                   avg_predicted_pd=("pd", "mean"), share_high_risk=("high", "mean"))
        t["adverse_impact_ratio"] = adverse_impact_ratio(t.share_high_risk)
        t.insert(0, "attribute", attribute)
        rows.append(t.reset_index())
    return pd.concat(rows, ignore_index=True)


def design_checks(book, X, y, train_idx, test_idx, params):
    """Re-test, on the hold-out, two design choices the documents rely on, with the logistic recipe:
    1. leaving age out (accuracy and the adverse impact ratio for each age band, with and without it);
    2. leaving collections out (does adding its counts and recoveries lift accuracy?)."""
    def fit(extra):
        Xe = X.assign(**{col: book[col].fillna(0).to_numpy() for col in extra})
        model = make_logistic(C=params["C"], numeric=params["numeric"] + extra)
        model.fit(Xe.iloc[train_idx], y.iloc[train_idx])
        return model.predict_proba(Xe.iloc[test_idx])[:, 1]

    def age_ratios(p):
        high = pd.Series(risk_band(p), index=test_idx).isin(["High", "Very High"])
        share = high.groupby(book.age_band.iloc[test_idx].to_numpy()).mean()
        return adverse_impact_ratio(share).round(3).to_dict()

    y_test = y.iloc[test_idx]
    p_base, p_age, p_coll = fit([]), fit(["age"]), fit(COLLECTIONS)
    return {
        "age": {"test_auc_without": roc_auc_score(y_test, p_base), "test_auc_with": roc_auc_score(y_test, p_age),
                "adverse_impact_ratio_test_without": age_ratios(p_base),
                "adverse_impact_ratio_test_with": age_ratios(p_age)},
        "collections": {"features": COLLECTIONS, "test_auc_without": roc_auc_score(y_test, p_base),
                        "test_auc_with": roc_auc_score(y_test, p_coll)},
    }


def main():
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    book = pd.read_csv(PROCESSED / "accounts_clean.csv")
    y = book.target_default
    categories = category_levels(book)
    X = model_frame(book, categories)
    groups = book.customer_id

    # 1. customer-grouped hold-out split
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
                               .split(X, y, groups))
    X_train, y_train, X_test, y_test = X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx], y.iloc[test_idx]
    print(f"train {len(train_idx):,} accounts | test {len(test_idx):,} accounts (no shared customers)")

    # 2. scorecard sign check, then tune both models with the same grouped cross-validation
    numeric, sign_dropped = sign_checked_features(X_train, y_train)
    print("tuning")
    best_lr, lr_grid = tune("logistic", [{**p, "numeric": numeric} for p in LOGISTIC_GRID], X_train, y_train,
                            groups.iloc[train_idx])
    best_gb, gb_grid = tune("lightgbm", LIGHTGBM_GRID, X_train, y_train, groups.iloc[train_idx])
    best = {"logistic": best_lr, "lightgbm": best_gb}

    # 3. fit on the training part, evaluate on the hold-out; 4. out-of-time check
    development = (book.origination_year <= 2021).to_numpy()
    recent_new = ~development & ~book.customer_id.isin(book.customer_id[development]).to_numpy()
    test_predictions = book.iloc[test_idx][["account_id", "customer_id", "target_default", "loan_type"]].copy()
    metrics = {}
    for kind in ["logistic", "lightgbm"]:
        model = build(kind, best[kind]["params"]).fit(X_train, y_train)
        p_train, p_test = model.predict_proba(X_train)[:, 1], model.predict_proba(X_test)[:, 1]
        test_predictions[f"pd_{kind}"] = p_test
        oot_model = build(kind, best[kind]["params"]).fit(X[development], y[development])
        p_oot = oot_model.predict_proba(X[recent_new])[:, 1]
        metrics[kind] = {
            "params": best[kind]["params"],
            "cv_auc_mean": best[kind]["cv_auc_mean"],
            "cv_auc_std": best[kind]["cv_auc_std"],
            "train": summarize(y_train, p_train),
            "test": summarize(y_test, p_test),
            "out_of_time": summarize(y[recent_new], p_oot),
            "psi_train_vs_test": psi(p_train, p_test),
            "psi_development_vs_recent": psi(oot_model.predict_proba(X[development])[:, 1], p_oot),
        }

    # 5. champion / challenger
    gain = metrics["lightgbm"]["test"]["auc"] - metrics["logistic"]["test"]["auc"]
    champion = "lightgbm" if gain >= MATERIALITY else "logistic"
    challenger = "logistic" if champion == "lightgbm" else "lightgbm"
    print(f"champion: {champion} (LightGBM test AUC gain {gain:+.3f}, needs >= {MATERIALITY})")

    # 6. refit both on all data and score the whole book
    bundles = {}
    for kind in ["logistic", "lightgbm"]:
        model = build(kind, best[kind]["params"]).fit(X, y)
        features = CATEGORICAL + numeric if kind == "logistic" else FEATURES
        bundle = {"kind": kind, "model": model, "categories": categories, "features": features,
                  "params": best[kind]["params"]}
        if kind == "logistic":
            bundle["background_mean"] = model.named_steps["prep"].transform(X).mean(axis=0)
        bundles[kind] = bundle
        joblib.dump(bundle, MODELS / f"{kind}.joblib")

    scores = pd.DataFrame({"account_id": book.account_id})
    for kind, bundle in bundles.items():
        scores[f"pd_{kind}"] = bundle["model"].predict_proba(X)[:, 1]
    scores["pd"] = scores[f"pd_{champion}"]
    scores["risk_band"] = risk_band(scores.pd)
    # share of the book with a strictly lower predicted PD: "riskier than x% of accounts"
    scores["pd_percentile"] = (scores.pd.rank(method="min") - 1) / len(scores)
    scores["lgd_assumed"] = book.loan_type.map(LGD)
    scores["expected_loss"] = scores.pd * scores.lgd_assumed * book.exposure
    contrib = contributions(bundles[champion], X)
    scores["top_risk_drivers"], scores["drivers_json"] = explain(book, contrib)
    scores.to_csv(PROCESSED / "scores.csv", index=False)

    # reports for the notebook, dashboard and documents
    test_predictions["pd"] = test_predictions[f"pd_{champion}"]
    test_predictions["risk_band"] = risk_band(test_predictions.pd)
    test_predictions.to_csv(REPORTS / "test_predictions.csv", index=False)
    decile_table(test_predictions.target_default, test_predictions.pd).to_csv(REPORTS / "deciles_test.csv",
                                                                              index=False)
    bands = test_predictions.groupby("risk_band").agg(accounts=("pd", "size"), avg_predicted_pd=("pd", "mean"),
                                                      actual_default_rate=("target_default", "mean"))
    bands = bands.reindex([name for _, name in RISK_BANDS])
    bands.reset_index().to_csv(REPORTS / "risk_bands_test.csv", index=False)

    importance = pd.DataFrame({"feature": FEATURES, "label": [LABELS[f] for f in FEATURES]})
    for kind, bundle in bundles.items():
        importance[f"mean_abs_shap_{kind}"] = importance.feature.map(contributions(bundle, X).abs().mean())
    importance = importance.sort_values(f"mean_abs_shap_{champion}", ascending=False)
    importance.to_csv(REPORTS / "feature_importance.csv", index=False)

    prep, lr = bundles["logistic"]["model"].named_steps["prep"], bundles["logistic"]["model"].named_steps["model"]
    pd.DataFrame({"term": prep.get_feature_names_out(), "coefficient": lr.coef_[0]}).sort_values(
        "coefficient", key=abs, ascending=False).to_csv(REPORTS / "logistic_coefficients.csv", index=False)

    fairness_table(book, scores).to_csv(REPORTS / "fairness.csv", index=False)

    # monitoring baseline: how much the inputs moved between older and newer vintages
    top_features = importance.sort_values(f"mean_abs_shap_{champion}", ascending=False).feature.head(10)
    csi = pd.DataFrame([{"feature": f, "label": LABELS[f],
                         "csi_development_vs_recent": psi(X.loc[development, f], X.loc[~development, f],
                                                          categorical=f in CATEGORICAL)}
                        for f in top_features])
    csi["status"] = csi.csi_development_vs_recent.map(psi_status)
    csi.to_csv(REPORTS / "csi_baseline.csv", index=False)

    summary = {
        "champion": champion,
        "challenger": challenger,
        "materiality_auc": MATERIALITY,
        "lightgbm_auc_gain_on_test": gain,
        "split": {"train_accounts": int(len(train_idx)), "test_accounts": int(len(test_idx)),
                  "oot_development_accounts": int(development.sum()), "oot_test_accounts": int(recent_new.sum())},
        "models": metrics,
        "tuning": lr_grid + gb_grid,
        "excluded_features": EXCLUDED,
        "logistic_features": {"numeric": numeric, "categorical": CATEGORICAL,
                              "dropped_by_sign_check": sign_dropped,
                              "left_out_as_duplicates": [f for f in FEATURES
                                                         if f not in EXPECTED_SIGN and f not in CATEGORICAL]},
        "risk_bands": RISK_BANDS,
        "design_checks": design_checks(book, X, y, train_idx, test_idx, best["logistic"]["params"]),
    }
    (REPORTS / "model_metrics.json").write_text(json.dumps(summary, indent=2))

    print("\n               AUC    Gini   KS     (hold-out test)   out-of-time AUC")
    for kind in ["logistic", "lightgbm"]:
        t, o = metrics[kind]["test"], metrics[kind]["out_of_time"]
        print(f"  {kind:<11} {t['auc']:.3f}  {t['gini']:.3f}  {t['ks']:.3f}                     {o['auc']:.3f}")
    print("\nrisk bands on the hold-out test set:")
    print(bands.round(3).to_string())
    checks = summary["design_checks"]
    print(f"\nage left out: test AUC {checks['age']['test_auc_without']:.4f} "
          f"(with age {checks['age']['test_auc_with']:.4f}); 21-25 adverse impact ratio "
          f"{checks['age']['adverse_impact_ratio_test_without']['21-25']:.3f} "
          f"(with age {checks['age']['adverse_impact_ratio_test_with']['21-25']:.3f})")
    print(f"collections left out: test AUC {checks['collections']['test_auc_without']:.4f} "
          f"(with collections {checks['collections']['test_auc_with']:.4f})")


if __name__ == "__main__":
    main()
