# Assumptions log

Anything invented or approximated, and what a real bank's IT/data team would need to provide to ship this.
Data quality fixes are detailed in `data_quality_notes.md`.

# Part A: Anything invented or approximated

## Data and definitions

| # | Assumption | Why | Impact if wrong |
|---|---|---|---|
| D1 | **As-of date = 2024-12-31.** The data has no snapshot date; the latest loan originates 2023-12-30 and every loan carries 12 months of repayment, ending by December 2024. | Makes every repayment window complete. | Trend and "recent" views shift; no effect on the model. |
| D2 | **`target_default` is observed after the 12-month repayment window.** Its definition and horizon are not given; no account reaches 90+ days past due inside the window. | Repayment features are then pre-default behaviour, not leakage. | If default were recorded inside the window, repayment features would leak and the model would be invalid. Needs the bank's default definition. |
| D3 | **Exposure = original loan amount.** Reported balances don't follow the loans' own payment schedules (12-month loans still show ~38% outstanding after the last payment). | Consistent and auditable. | Overstates exposure on seasoned loans; EAD needs real balances. |
| D4 | **Bureau data is a current snapshot, not as of each application.** 68% of lines open after the customer's first loan; some close dates are in 2029. Bureau dates are not used in any feature. | Filtering to lines opened before each loan kept 37% of lines and weakened the signal. | Point-in-time leakage risk in production; needs archived bureau pulls. |
| D5 | **Collections data is excluded from the model** and shown with a caveat. 45% of events are dated before the loan existed, and outcomes contradict amounts. | Unreliable, and adds no lift (hold-out AUC 0.6873 without, 0.6868 with; re-tested in every training run). | Collections history could add signal once its dates are trustworthy. |
| D6 | **`customer_since` is unreliable** (later than the customer's own first loan for 57% of customers). Relationship start = the earlier of the two dates. | Avoids impossible tenures. | Relationship-tenure analysis not possible until fixed at source. |
| D7 | **Credit cards are treated as fixed-tenure instalment loans**, as the data gives them a tenure and a fixed instalment. | Follows the data. | Real cards are revolving; they need limit, utilisation and statement data and usually their own scorecard. |
| D8 | **294 missing interest rates were back-solved** from the instalment, amount and tenure with the standard loan formula (error under 0.011 percentage points on known rates). | Recovers the true value instead of guessing. | Negligible. |
| D9 | **Imputation and caps:** medians for missing years employed (by employment type), dependents and years at address; years employed capped at age − 16. Medians are computed on the whole dataset. | Missingness looks random (5%, overlaps at chance level). | Negligible leakage into validation; production learns medians on training data only. |
| D10 | **Loan purpose is unreliable** (statistically unrelated to product; e.g. mortgages for "Vehicle Purchase"). Shown, not modelled. | Avoids learning noise. | None for the model. |
| D11 | **Geography and regulation:** amounts are in USD but the vocabulary is Indian banking (NBFC, DSA, EMI, Tier 1/2/3 cities). The deck uses a **US regulatory frame** (SR 11-7 model risk guidance, ECOA / Regulation B fair lending). | A single, explicit frame. | For an Indian bank, map the same controls to RBI guidance on credit model governance and its Fair Practices Code. |
| D12 | **The data is synthetic.** Default rates are flat across vintages and there is no drift (PSI < 0.01). | Given. | Real books drift; monitoring thresholds matter more in practice. |

## Modelling

| # | Assumption | Note |
|---|---|---|
| M1 | **Loss-given-default is illustrative:** Mortgage 20%, Auto 40%, Business 60%, Personal 65%, Credit card 75%. Used only for the expected-loss figure. | The data has no recovery or collateral values. |
| M2 | **Risk band cut-offs** (PD < 10% Low, 10–20% Medium, 20–35% High, 35%+ Very High) were chosen for readability. | In production they are set by credit policy and risk appetite. |
| M3 | **Champion rule:** LightGBM had to beat logistic regression by 0.01 test AUC to justify its lower transparency. It scored 0.009 lower. | A materiality threshold, agreed with model risk in practice. |
| M4 | **Protected attributes are excluded:** gender, marital status and age. Age cost no accuracy to remove (hold-out AUC 0.685 with it, 0.687 without; re-tested in every training run). Region and city tier are kept as business segments. | Geography can proxy for protected groups; it should be in the fair-lending review. |
| M5 | **Remaining age gap:** borrowers aged 21–25 still have an adverse impact ratio of 0.72 through income and job tenure. | Flagged for Compliance review, not hidden. |
| M6 | **Scorecard sign check:** the logistic model keeps one feature per idea, and training dropped 4 features whose coefficient pointed the wrong way (overlap with other features). | Keeps per-account reasons trustworthy; costs no accuracy. |

## AI feature (and bonus AI)

| # | Assumption | Note |
|---|---|---|
| A1 | **Anomaly thresholds are first guesses:** 5% flagged as peer outliers, 1% High severity, peer group = loan type x income band (minimum 50 accounts). | Should be tuned with analysts' confirm / dismiss decisions after a pilot. |
| A2 | **"Own history" is limited to 12 months** of repayment, so the payment-shock rule compares the last 3 months with the first 9. | Longer histories allow better baselines. |
| A3 | **The chat uses NVIDIA's hosted API in this prototype.** Tool results (aggregates and, for account questions, one account's details) leave the machine. Acceptable only because the data is synthetic. | Production: run Nemotron as an NVIDIA NIM inside the bank's network so no customer data reaches a third party. |
| A4 | **Model choice:** Nemotron-3 Ultra passed a tool-calling test (correct tool, about 2 seconds); Nemotron-3.5 Lightning is the fallback. Several other hosted models were unavailable or timed out for this key. | Re-evaluate on the bank's own question set before production. |
| A5 | **Review decisions and chat ratings are logged to CSV files** in `data/review/`. | Production needs a database with an audit trail and user identity from single sign-on. |
| A6 | **The chat's number check has a known limit:** it confirms each number exists in the tool results and isn't placed next to a segment it doesn't belong to, but it can't catch a real number described too broadly (e.g. "every region defaults at 16.1%", the portfolio rate). It is also strict: a number the model works out itself (a difference of two rates) is flagged even when the arithmetic is right. | This is why every answer shows its queries and the source of each number, and the analyst checks before a figure is used. |

# Part B: What a real bank's IT/data team would need to ship this

| Area | Needed | Why |
|---|---|---|
| Core banking | Loan master with **current balances**, transaction-level payments and monthly account snapshots with as-of dates | Real exposure (EAD), longer repayment history, proper vintage curves |
| Risk data | The bank's **default definition** and observation / performance windows; write-off and recovery history; **collateral values (LTV)** for mortgages and autos | Correct target, real LGD and expected loss, mortgage risk sizing |
| Credit bureau | **Archived point-in-time bureau pulls** (at application and refreshed monthly), bureau scores and enquiries, plus the data-sharing agreement | Removes the snapshot leakage risk (D4) |
| Collections | System extract with reliable event dates linked to accounts | Lets collections history be used (D5) |
| Customer data | CRM customer master (relationship start), verified income and employment | Fixes D6, supports income verification for flagged groups |
| Platform | Scheduler (e.g. Airflow) for monthly runs, model registry (e.g. MLflow), container hosting for the dashboard behind single sign-on with role-based access, GPU hosting for an NVIDIA NIM inside the network, central logging | Turns the prototype into a governed service |
| Governance | Independent model validation (SR 11-7), Compliance fair-lending review and **adverse-action reason codes** (from the per-account drivers), Information Security review of the LLM, data privacy sign-off | Required before any model output influences decisions |
| Business | Credit policy owners for band cut-offs and the proposed debt-to-income cap; a pilot group of analysts for the anomaly queue | Thresholds, adoption and feedback |
