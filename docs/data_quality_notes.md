# Data Quality Notes

Produced by `src/ingest.py`. Every count below is computed by the code and saved in
`data/processed/dq_log.csv`, so rerunning the pipeline regenerates it.

## Short notes

- **Join logic and keys:** `loan_accounts` is the spine (one row per loan, 14,685 rows). `customers` joins on
  `customer_id`; `bureau_credit` (22,979 lines) and `repayment_history` (176,220 rows) are first aggregated to one row
  per customer and per account, then joined on `customer_id` and `account_id`; `collections_events` is aggregated per
  account for the dashboard only. Every join is validated and the row count stays at 14,685.
- **Missing values, and how:** years employed (500): 0 for the 23 unemployed customers, median of the same employment
  type for the rest, flagged. Dependents and years at address (500 each): median, flagged. Interest rate (294):
  **recovered exactly** from the instalment, amount and tenure. Bureau days past due (689): 0 on closed lines, median of
  the same status otherwise, flagged. 1,008 customers with no bureau record: flagged as thin file, not filled with zeros.
- **Inconsistent labels:** gender coded "M" / "F" for 600 customers, mapped to "Male" / "Female".
- **Other issues:** 726 impossible job tenures (capped at age − 16); `customer_since` later than the customer's own
  first loan for 57% of customers (not used); bureau dates after the as-of date (flagged, not used); 45% of collections
  events dated before their loan existed (collections kept out of the model); balances that don't follow the payment
  schedule (exposure = original loan amount); 79 loans whose instalment exceeds the borrower's income (flagged).
- **Nothing was silently dropped:** every row is kept, and all 45 checks are logged with the rows they affected.

The sections below are the full detail behind these notes.

## 1. Source tables

| Table | Rows | One row per | Key | Links to |
|---|---|---|---|---|
| customers | 10,000 | customer | `customer_id` | — |
| loan_accounts | 14,685 | loan account | `account_id` | customers via `customer_id` |
| bureau_credit | 22,979 | external credit line | `bureau_id` | customers via `customer_id` |
| repayment_history | 176,220 | account × month (months 1–12) | `account_id` + `payment_month` | loan_accounts via `account_id` |
| collections_events | 8,635 | collections action | `collection_event_id` | loan_accounts via `account_id` |

## 2. Join logic

```
customers ──1:N── loan_accounts ──1:12── repayment_history
    │                   └──────1:N────── collections_events
    └──1:N── bureau_credit
```

The output has **one row per loan account (14,685 rows)**, because that is where the target
`target_default` lives.

1. Start from `loan_accounts`.
2. Add the customer profile: many-to-one on `customer_id`.
3. Summarise each customer's bureau lines (counts, write-offs, days past due, utilisation), then join
   many-to-one on `customer_id`. All loans of the same customer get the same bureau summary.
4. Summarise each account's 12 months of repayment (missed and late counts, days past due, share of
   the instalment paid, longest missed streak, early vs recent behaviour), then join one-to-one on `account_id`.
5. Summarise each account's collections events (valid events only), then join one-to-one on
   `account_id`. These columns feed the dashboard, not the model.

Raw tables are never joined directly. A customer with 3 loans and 5 bureau lines would become
3 × 5 × 12 = 180 rows and the same exposure would be counted many times. Every join is validated
(pandas `validate=`) and the row count is asserted to stay at 14,685.

## 3. As-of date: 2024-12-31

The data has no snapshot date. Loans originate between 2018-01-01 and 2023-12-30, and each carries
12 months of repayment history, so the last repayment month ends in December 2024. Choosing
2024-12-31 makes every repayment window complete. Bureau dates after it are treated as errors.

## 4. Issues found and how each was handled

### customers (10,000 rows)

| Issue | Rows | How it was handled |
|---|---|---|
| Gender coded "M" / "F" instead of "Male" / "Female" | 600 (6.0%) | Mapped to the full labels |
| `years_employed` missing for Unemployed customers | 23 | Set to 0: every other Unemployed customer has 0, so this gap is structural, not random |
| `years_employed` longer than age − 16 (e.g. a 25-year-old with 15 years in the job). The column has zero correlation with age | 726 (7.3%) | Capped at age − 16, flagged `years_employed_capped` |
| `years_employed` missing for employed customers | 477 (4.8%) | Median of the same employment type, flagged |
| `num_dependents` missing | 500 (5.0%) | Median (1), flagged |
| `years_at_residence` missing | 500 (5.0%) | Median (2.8 years), flagged |
| `years_at_residence` longer than the customer's age | 5 | Capped at age |
| Income exactly $12,000, the minimum (values below look clipped) | 76 (0.8%) | Kept, flagged `income_at_floor` |
| Unemployed but earning more than the employed median ($80,700) | 22 | Kept (could be pension or rental income); listed for income verification |
| `customer_since` is later than the customer's own first loan | 5,673 customers (56.7%) | Field judged unreliable. `relationship_start` = earlier of the two dates. Not used in the model |

### loan_accounts (14,685 rows)

| Issue | Rows | How it was handled |
|---|---|---|
| `interest_rate_pct` missing | 294 (2.0%) | **Recovered exactly instead of guessed.** `emi_due` follows the standard EMI formula, so the rate that produces each loan's observed instalment was solved numerically. On the 14,391 loans with a known rate, this method reproduces the rate within 0.011 percentage points. Flagged `interest_rate_derived` |
| Monthly instalment larger than the borrower's whole monthly income | 79 (0.5%) | Kept (the exposure is real), flagged `emi_exceeds_income`. These loans would fail any affordability check; **45.6% of them defaulted** |
| Loan purpose doesn't fit the product (a mortgage for "Vehicle Purchase", an auto loan for "Wedding"). Purpose is statistically unrelated to product (Cramér's V = 0.02) | 4,475 (30.5%) | Kept for display, flagged, left out of the model |
| Loan amount exactly $1,000, the minimum (values below look clipped) | 292 (2.0%) | Kept, flagged |
| `collateral_flag` disagrees with loan type | 0 | Check passed, but the flag only repeats loan type (Mortgage and Auto = 1), so it is left out of the model |
| Channel label "Partner/DSA" differs from the dictionary's "Partner-DSA" | — | Data label kept |

### bureau_credit (22,979 rows)

| Issue | Rows | How it was handled |
|---|---|---|
| Customer has no bureau record at all (thin file) | 1,008 customers (10.1%) | `bureau_no_file` = 1; counts set to 0; days-past-due and ratio features left blank, because "no history" is not the same as "clean history" |
| Days past due missing on Closed lines | 260 | Set to 0: every other Closed line reports 0 |
| Days past due missing on Active / Written-Off lines | 429 | Median of the same status, flagged |
| `date_opened` after the as-of date | 2,898 (12.6%) | Flagged; bureau dates are not used in any feature |
| `date_closed` after the as-of date, some as late as 2029-12-24 | 4,673 (20.3%) | Flagged; bureau dates are not used in any feature |
| Line opened after the customer's first loan, so the table is a current snapshot rather than a pull as of each application | 15,523 (67.6%) | Used as a current snapshot. Keeping only lines opened before each loan leaves 37% of lines and weakens both bureau signals tested. **Logged as a leakage risk**: production needs bureau data as of each application date |

### repayment_history (176,220 rows)

| Issue | Rows | How it was handled |
|---|---|---|
| Month-12 balance doesn't follow the loan's own payment schedule (12-month loans still show ~38% outstanding after their final instalment) | 4,715 of the 8,573 accounts that never missed a payment (55%) | Balance not used as exposure. **Exposure = original loan amount** |

### collections_events (8,635 rows)

| Issue | Rows | How it was handled |
|---|---|---|
| Event dated before the loan existed | 3,875 (44.9%) | Excluded from collections metrics, flagged |
| "Payment Received" with $0 recovered | 1,287 (14.9%) | Flagged |
| "No Response" with money recovered | 443 (5.1%) | Flagged |
| "Written-Off Referral" on a loan marked as performing | 1,159 (13.4%) | Counted only. A flag built from the target is not stored, so it can't leak into a model |
| Recovered more than the original loan amount | 95 (1.1%) | Flagged |
| Action label "SMS/Email Notice" differs from the dictionary's "SMS-Email" | — | Data label kept |
| Whole table | 8,635 | **Dashboard only, left out of the model**: the dates are unreliable, and adding collections features gives no lift (hold-out AUC 0.6873 without, 0.6868 with; re-tested in every training run) |

## 5. Checks that passed

- No duplicate keys in any table, and no orphan records: every loan has a customer, and every
  repayment and collections row has a loan.
- Every account has exactly months 1 to 12 of repayment history.
- Payment status always agrees with days past due and amount paid: On-Time = 0 days and paid in full,
  Late = 1–29 days and paid in full, Missed = 30+ days and nothing paid.
- The instalment is constant within each account and matches the standard EMI formula.
- Bureau status, balance and close date always agree. No balance exceeds its limit and no line closes
  before it opens.
- **Leakage check:** no account reaches 90+ days past due inside the 12-month window, so the default
  itself is not visible in the repayment features. They describe behaviour *before* default.

## 6. Missing values look random

The three customer fields are each missing for exactly 500 customers (5%). They overlap no more than
chance predicts (about 25 customers per pair), and default rates for imputed rows sit within about
3 percentage points of the rest. The "was imputed" flags are kept so the model can use them if they
matter.

## 7. Decisions carried into analysis and modelling

- **Exposure = original loan amount**, because reported balances don't reconcile with the payment schedules.
- **The model uses** application, bureau and repayment features. **It leaves out** collections,
  `loan_purpose`, `collateral_flag` and `customer_since` for the reasons above, and gender for
  fair-lending reasons.
- Imputation medians are computed on the whole dataset. With 5% random missingness the effect on
  validation scores is negligible; a production pipeline would learn them on training data only.

## 8. Output files (`data/processed/`)

| File | Contents |
|---|---|
| `accounts_clean.csv` | **The analysis-ready dataset**: 14,685 rows (one per loan) × 85 columns |
| `customers_clean.csv`, `bureau_clean.csv`, `repayment_clean.csv`, `collections_clean.csv` | Cleaned detail tables with their flags, used for dashboard drill-downs |
| `dq_log.csv` | Every check run: table, column, issue, rows affected, action taken |

Column groups in `accounts_clean.csv`:

- **IDs and target:** `account_id`, `customer_id`, `target_default`
- **Loan:** type, purpose, amount, exposure, interest rate, tenure, EMI, origination date / year / quarter, channel
- **Customer:** age, gender, marital status, education, employment, sector, years employed, income,
  region, city tier, dependents, housing, years at residence, plus `age_band` and `income_band`
- **Bureau summary (`bureau_*`):** number of lines (all, active, closed, written-off, 30+ days past due),
  any write-off, write-off share, max and mean days past due, total sanctioned and outstanding,
  active utilisation, share of non-bank lenders, no-file flag
- **Repayment summary (`rep_*`):** on-time / late / missed counts, max and mean days past due,
  mean days past due in the first vs last 3 months and the trend between them, missed in the last
  3 months, share of instalments paid, longest missed streak, months since last missed payment,
  ever 30+ / 60+ days past due, month-12 balance
- **Collections summary (`coll_*`, dashboard only):** valid and raw event counts, amount recovered,
  last action, outcome and date
- **Ratios:** instalment-to-income, loan-to-income, bureau debt-to-income, loans per customer, customer total exposure
- **Data quality flags:** `*_imputed`, `*_capped`, `*_at_floor`, `interest_rate_derived`,
  `purpose_product_mismatch`, `customer_since_after_first_loan`, `emi_exceeds_income`
