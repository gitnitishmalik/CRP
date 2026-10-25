# Product brief: Credit Risk Portfolio Intelligence

A portfolio risk workspace that shows where a lending book's risk and losses sit, scores every account with an
explainable default model, and lets the risk team question the data in plain English, with every answer grounded in
the real numbers.

## Problem

The Chief Risk Officer's team sees the loan book through static monthly reports, which leaves three gaps:

1. **Concentration hides in averages.** The book's default rate is 16.1%, yet mortgages, 10% of accounts, hold 65% of
   defaulted exposure, and loans whose instalment exceeds half the borrower's income default at 30.8%.
2. **No account-level early warning.** Risk is managed by segment; analysts can't see which accounts are slipping, or
   why, until collections starts.
3. **Ad-hoc questions take days.** "Which segment has the highest default rate this quarter?" goes into an analyst
   queue and comes back as a spreadsheet nobody can trace to the data.

## Target user

| User | Need | Uses |
|---|---|---|
| **Chief Risk Officer** and credit committee (primary) | Where risk concentrates; whether the model can be trusted; which policies to change | Portfolio view, model page, pitch deck |
| **Portfolio risk analysts** (daily users) | Find and explain risky segments and accounts; answer questions fast | Segment drill-down, account-level view, Ask the Portfolio |
| **Model risk and Compliance** (reviewers) | Validate performance, stability and fairness | Model page, notebooks, assumptions log |

## Scope

| In scope | Out of scope |
|---|---|
| Ingestion and cleaning of the 5 source tables into one analysis-ready dataset, with logged data quality checks | Automated credit decisions or customer contact: every AI output goes to a person first |
| Portfolio profiling, cross-cuts and 5 insights with policy implications | Application (approval) scoring: the model uses 12 months of repayment, so it scores accounts already on the book |
| Default model (logistic regression vs LightGBM) scoring every account, with key drivers, fairness and monitoring | Loss models (LGD, EAD) and IFRS 9 / CECL provisions; expected loss shown is illustrative |
| Interactive dashboard: portfolio view, account-level view, filters and drill-down | Live data feeds, single sign-on and role-based access, production deployment |
| AI feature: "Ask the Portfolio" natural-language Q&A, grounded in the data (plus a bonus anomaly-review queue) | Forecasting future default rates; the data is historical only |
