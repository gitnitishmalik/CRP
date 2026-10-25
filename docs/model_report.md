# Model report

Full detail with charts: `notebooks/03_model_building.ipynb`. Code: `src/features.py`, `src/train_model.py`.

## Target and use

Predicts `target_default` (16.1% of 14,685 loan accounts) from the application, the customer's bureau history and the
account's first 12 months of repayment. Default is not visible inside that window (no account reaches 90 days past
due), so the repayment features describe behaviour before default. This is a **behavioural score for accounts already
on the book**: an early-warning tool, not an approval model.

## Features (engineered, not raw)

- **Bureau, aggregated per customer:** active lines, written-off lines, lines 30+ days late, average days past due,
  bureau debt-to-income, share of credit from non-bank lenders.
- **Repayment, aggregated per account:** late and missed payments, worst days past due, missed payments in the last 3
  months.
- **Application and affordability:** income, years in job, years at address, dependents, loan-to-income, interest rate,
  number of loans held, plus loan type, employment, education, region and other categories.
- **Scorecard sign check:** one feature per idea, each with an expected direction; training dropped 4 features whose
  coefficients pointed the wrong way (`bureau_active_utilization`, `bureau_no_file`, `rep_months_since_missed`,
  `rep_max_missed_streak`). LightGBM uses all 40 features with monotone constraints instead.
- **Deliberately excluded:** gender, marital status and age (protected characteristics), loan purpose (unreliable),
  collateral flag (repeats loan type), customer-since date (unreliable), collections data (unreliable dates; adding
  its counts and recoveries gives no lift: hold-out AUC 0.6873 without, 0.6868 with).

## Validation and results

| | Logistic regression (champion) | LightGBM (challenger) |
|---|---|---|
| 5-fold grouped CV AUC | 0.672 ± 0.007 | 0.668 ± 0.005 |
| **Hold-out test AUC** (2,948 accounts, unseen customers) | **0.687** | 0.678 |
| Test Gini | 0.375 | 0.356 |
| Test KS | 0.288 | 0.277 |
| Brier score | 0.1245 | 0.1259 |
| Out-of-time AUC (train 2018–21, test 2022–23 new customers) | 0.700 | 0.695 |
| Train AUC (overfitting check) | 0.685 | 0.763 |

**Champion: logistic regression.** LightGBM needed +0.01 test AUC to justify lower transparency; it scored 0.009 lower
and overfits. The signal here is additive and monotonic, so the linear model captures it, and its per-account
explanations are exact (coefficient x deviation from average), which supports adverse-action reason codes.

**Ranking and calibration (test):** the riskiest 10% default at 38.6% (2.4x average); the riskiest 30% hold 54% of
defaults. Predicted vs actual default rate by band: Low 7.7% vs 6.8%, Medium 14.2% vs 14.4%, High 25.4% vs 26.1%, Very
High 45.5% vs 41.9%.

**What drives the score (mean absolute SHAP, champion):** late payments, missed payments, annual income, years in
current job, loan-to-income, bureau active lines (more lines lower risk), bureau average days past due.

## Fairness

None of gender, marital status or age is an input. Adverse impact ratio (share not rated High / Very High vs the most
favoured group, four-fifths rule):

| Group | Ratio | |
|---|---|---|
| Gender (F / M) | 0.994 / 1.000 | pass |
| Marital status | 0.973 to 1.000 | pass |
| Age 21–25 / 26–35 | 0.716 / 0.796 | review |

Adding age back would lower AUC slightly (0.687 to 0.685) and push the 21–25 ratio on the test set from 0.73 down to
0.53, so it stays out. Training re-runs both of these design checks every time (`design_checks` in
`reports/model_metrics.json`). The remaining gap for younger borrowers runs through income and job tenure; it is
documented for Compliance review.

## Monitoring after deployment

After go-live the model would be monitored on three clocks. **Monthly**, the population stability index (PSI) of the
score distribution and the characteristic stability index (CSI) of the top ten features are compared with the
development baseline (today: score PSI 0.003 train vs test, 0.007 between 2018–21 and 2022–23 vintages; all CSI below
0.003): under 0.10 is stable, 0.10–0.25 triggers an investigation, above 0.25 escalates to model risk management.
**Quarterly**, as outcomes mature, AUC, KS and calibration by risk band are re-measured against the validated baseline;
a relative Gini drop of more than 10%, or a gap of more than 20% between predicted and actual default rates in any
band, triggers a review. The model is **retrained annually**, or earlier on any breach, with LightGBM re-run as a
challenger each time. Fairness ratios and AI review outcomes (chat ratings, anomaly decisions) are re-checked quarterly, and every change goes
through independent validation in line with SR 11-7.
