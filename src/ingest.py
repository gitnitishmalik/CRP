"""
Step 1 - Data ingestion and cleaning.

Loads the 5 raw tables, fixes data quality issues, aggregates the one-to-many tables
and joins everything into one analysis-ready table with one row per loan account.
Every issue found is counted and written to dq_log.csv, including checks that passed.

    python src/ingest.py

Join logic (loan_accounts is the spine; every join is validated so no rows multiply)
    loan_accounts       14,685 rows, 1 per loan account, holds target_default
      + customers           on customer_id   many-to-one
      + bureau_credit       on customer_id   aggregated to 1 row per customer first
      + repayment_history   on account_id    aggregated to 1 row per account first
      + collections_events  on account_id    aggregated to 1 row per account first (dashboard only)
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from config import AS_OF_DATE, MIN_WORKING_AGE, PROCESSED, RAW


class DQLog:
    """Collects every data quality finding with the number of rows it touched."""

    def __init__(self):
        self.rows = []

    def add(self, table, column, issue, n, total, action):
        self.rows.append({
            "table": table,
            "column": column,
            "issue": issue,
            "rows_affected": int(n),
            "pct_of_rows": round(100 * n / total, 2),
            "action": action,
        })

    def to_frame(self):
        return pd.DataFrame(self.rows)


def load_raw():
    customers = pd.read_csv(RAW / "customers.csv", parse_dates=["customer_since"])
    loans = pd.read_csv(RAW / "loan_accounts.csv", parse_dates=["origination_date"])
    bureau = pd.read_csv(RAW / "bureau_credit.csv", parse_dates=["date_opened", "date_closed"])
    repayment = pd.read_csv(RAW / "repayment_history.csv")
    collections = pd.read_csv(RAW / "collections_events.csv", parse_dates=["event_date"])
    return customers, loans, bureau, repayment, collections


def check_keys(customers, loans, bureau, repayment, collections, log):
    checks = [
        ("customers", "customer_id", "duplicate primary key",
         customers.customer_id.duplicated().sum(), len(customers)),
        ("loan_accounts", "account_id", "duplicate primary key",
         loans.account_id.duplicated().sum(), len(loans)),
        ("bureau_credit", "bureau_id", "duplicate primary key",
         bureau.bureau_id.duplicated().sum(), len(bureau)),
        ("collections_events", "collection_event_id", "duplicate primary key",
         collections.collection_event_id.duplicated().sum(), len(collections)),
        ("repayment_history", "account_id + payment_month", "same month recorded twice",
         repayment.duplicated(["account_id", "payment_month"]).sum(), len(repayment)),
        ("loan_accounts", "customer_id", "customer missing from customers table",
         (~loans.customer_id.isin(customers.customer_id)).sum(), len(loans)),
        ("bureau_credit", "customer_id", "customer missing from customers table",
         (~bureau.customer_id.isin(customers.customer_id)).sum(), len(bureau)),
        ("repayment_history", "account_id", "account missing from loan_accounts",
         (~repayment.account_id.isin(loans.account_id)).sum(), len(repayment)),
        ("collections_events", "account_id", "account missing from loan_accounts",
         (~collections.account_id.isin(loans.account_id)).sum(), len(collections)),
    ]
    for table, column, issue, n, total in checks:
        log.add(table, column, issue, n, total, "check passed" if n == 0 else "FAILED")
    # every join below assumes clean keys, so stop here rather than build on broken ones
    assert all(check[3] == 0 for check in checks), "key integrity problem - see dq_log.csv"


# ---------------------------------------------------------------- customers

def clean_customers(customers, loans, log):
    c = customers.copy()
    total = len(c)

    short_label = c.gender.isin(["M", "F"])
    c["gender"] = c.gender.replace({"M": "Male", "F": "Female"})
    log.add("customers", "gender", "inconsistent labels 'M' / 'F'", short_label.sum(), total,
            "mapped to 'Male' / 'Female'")

    # years_employed: structural zero first, then cap impossible values, then impute
    unemployed_missing = c.employment_type.eq("Unemployed") & c.years_employed.isna()
    c.loc[unemployed_missing, "years_employed"] = 0
    log.add("customers", "years_employed", "missing for Unemployed customers", unemployed_missing.sum(), total,
            "set to 0 (every other Unemployed customer has 0)")

    max_years = c.age - MIN_WORKING_AGE
    too_long = c.years_employed > max_years
    c["years_employed_capped"] = too_long.astype(int)
    c.loc[too_long, "years_employed"] = max_years[too_long]
    log.add("customers", "years_employed", f"longer than age - {MIN_WORKING_AGE} (impossible work history)",
            too_long.sum(), total, f"capped at age - {MIN_WORKING_AGE}, flagged")

    missing = c.years_employed.isna()
    median_by_type = c.groupby("employment_type").years_employed.transform("median")
    c["years_employed_imputed"] = missing.astype(int)
    c["years_employed"] = c.years_employed.fillna(np.minimum(median_by_type, max_years))
    log.add("customers", "years_employed", "missing", missing.sum(), total,
            "median of the same employment type, flagged")

    for col in ["num_dependents", "years_at_residence"]:
        missing = c[col].isna()
        fill = c[col].median()
        c[f"{col}_imputed"] = missing.astype(int)
        c[col] = c[col].fillna(fill)
        log.add("customers", col, "missing", missing.sum(), total, f"median ({fill:g}), flagged")

    too_long = c.years_at_residence > c.age
    c.loc[too_long, "years_at_residence"] = c.age[too_long]
    log.add("customers", "years_at_residence", "longer than the customer's age", too_long.sum(), total,
            "capped at age")

    floor = c.annual_income.min()
    at_floor = c.annual_income.eq(floor)
    c["income_at_floor"] = at_floor.astype(int)
    log.add("customers", "annual_income", f"sits exactly at the {floor:,.0f} minimum (lower values look clipped)",
            at_floor.sum(), total, "kept, flagged")

    employed_median = c.loc[c.employment_type.ne("Unemployed"), "annual_income"].median()
    odd_income = c.employment_type.eq("Unemployed") & (c.annual_income > employed_median)
    log.add("customers", "annual_income", f"Unemployed but earning above the employed median ({employed_median:,.0f})",
            odd_income.sum(), total, "kept (could be pension or rental income); needs income verification")

    first_loan = loans.groupby("customer_id").origination_date.min()
    c["first_loan_date"] = c.customer_id.map(first_loan)
    since_after_loan = c.customer_since > c.first_loan_date
    c["customer_since_after_first_loan"] = since_after_loan.astype(int)
    c["relationship_start"] = c[["customer_since", "first_loan_date"]].min(axis=1)
    log.add("customers", "customer_since", "later than the customer's own first loan", since_after_loan.sum(), total,
            "unreliable: relationship_start = earlier of the two dates; not used in the model")

    c["age_band"] = pd.cut(c.age, [20, 25, 35, 45, 55, 120],
                           labels=["21-25", "26-35", "36-45", "46-55", "56+"]).astype(str)
    c["income_band"] = pd.cut(c.annual_income, [0, 30_000, 50_000, 75_000, 100_000, np.inf], right=False,
                              labels=["<30k", "30-50k", "50-75k", "75-100k", "100k+"]).astype(str)
    return c


# ---------------------------------------------------------------- loan accounts

def annuity_rate(principal, months, emi, iterations=60):
    """Annual interest rate (%) at which the standard EMI formula returns `emi` (vectorised bisection)."""
    low = np.full(len(principal), 1e-9)
    high = np.full(len(principal), 0.2)
    for _ in range(iterations):
        mid = (low + high) / 2
        too_high = principal * mid / (1 - (1 + mid) ** -months) > emi
        high = np.where(too_high, mid, high)
        low = np.where(too_high, low, mid)
    return (low + high) / 2 * 1200


def clean_loans(loans, repayment, log):
    l = loans.copy()
    total = len(l)
    l["emi"] = l.account_id.map(repayment.groupby("account_id").emi_due.first())

    # emi_due follows the EMI formula exactly, so a missing rate can be solved for instead of guessed
    known = l.interest_rate_pct.notna()
    solved = pd.Series(annuity_rate(l.loan_amount.to_numpy(), l.tenure_months.to_numpy(), l.emi.to_numpy()),
                       index=l.index)
    max_error = (solved[known] - l.interest_rate_pct[known]).abs().max()
    l["interest_rate_derived"] = (~known).astype(int)
    l.loc[~known, "interest_rate_pct"] = solved[~known].round(2)
    log.add("loan_accounts", "interest_rate_pct", "missing", (~known).sum(), total,
            f"back-solved from emi_due, loan_amount and tenure with the EMI formula "
            f"(max error on known rows: {max_error:.3f} pct points), flagged")

    secured = l.loan_type.isin(["Mortgage", "Auto Loan"]).astype(int)
    log.add("loan_accounts", "collateral_flag", "disagrees with loan_type (Mortgage / Auto Loan = secured)",
            (secured != l.collateral_flag).sum(), total,
            "check passed; the flag only repeats loan_type, so it is left out of the model")

    floor = l.loan_amount.min()
    at_floor = l.loan_amount.eq(floor)
    l["loan_amount_at_floor"] = at_floor.astype(int)
    log.add("loan_accounts", "loan_amount", f"sits exactly at the {floor:,.0f} minimum (lower values look clipped)",
            at_floor.sum(), total, "kept, flagged")

    plausible = {
        "Auto Loan": {"Vehicle Purchase"},
        "Mortgage": {"Home Improvement", "Debt Consolidation", "Other"},
        "Business Loan": {"Business Expansion", "Debt Consolidation", "Other"},
    }
    mismatch = [t in plausible and p not in plausible[t] for t, p in zip(l.loan_type, l.loan_purpose)]
    l["purpose_product_mismatch"] = np.array(mismatch, dtype=int)
    chi2 = chi2_contingency(pd.crosstab(l.loan_purpose, l.loan_type))[0]
    cramers_v = np.sqrt(chi2 / (total * (min(l.loan_purpose.nunique(), l.loan_type.nunique()) - 1)))
    log.add("loan_accounts", "loan_purpose",
            f"does not fit the product, e.g. a Mortgage for 'Vehicle Purchase' "
            f"(purpose is unrelated to product: Cramer's V = {cramers_v:.3f})",
            l.purpose_product_mismatch.sum(), total, "kept for display, flagged, left out of the model")

    l["origination_year"] = l.origination_date.dt.year
    l["origination_quarter"] = l.origination_date.dt.to_period("Q").astype(str)
    window_end = l.origination_date + pd.DateOffset(months=12)
    past_last_loan = window_end > l.origination_date.max()
    log.add("loan_accounts", "origination_date",
            "12-month repayment window ends after the last loan date (data has no snapshot date)",
            past_last_loan.sum(), total, f"as-of date set to {AS_OF_DATE.date()} so every window is complete")
    return l


# ---------------------------------------------------------------- bureau

def clean_bureau(bureau, customers, loans, log):
    b = bureau.copy()
    total = len(b)

    no_file = ~customers.customer_id.isin(b.customer_id)
    log.add("bureau_credit", "customer_id", "customer has no bureau record (thin file)", no_file.sum(),
            len(customers), "bureau_no_file = 1; counts set to 0, DPD and ratio features left blank")

    closed_inconsistent = b.account_status.eq("Closed") & (b.outstanding_amount.ne(0) | b.date_closed.isna())
    active_inconsistent = b.account_status.eq("Active") & b.date_closed.notna()
    log.add("bureau_credit", "account_status", "status disagrees with balance or close date",
            (closed_inconsistent | active_inconsistent).sum(), total, "check passed")
    log.add("bureau_credit", "outstanding_amount", "above the sanctioned amount",
            (b.outstanding_amount > b.sanctioned_amount).sum(), total, "check passed")
    log.add("bureau_credit", "date_closed", "before date_opened",
            (b.date_closed < b.date_opened).sum(), total, "check passed")

    closed_missing = b.account_status.eq("Closed") & b.days_past_due.isna()
    b.loc[closed_missing, "days_past_due"] = 0
    log.add("bureau_credit", "days_past_due", "missing on Closed lines", closed_missing.sum(), total,
            "set to 0 (every other Closed line reports 0)")
    other_missing = b.days_past_due.isna()
    b["dpd_imputed"] = (closed_missing | other_missing).astype(int)
    b["days_past_due"] = b.days_past_due.fillna(b.groupby("account_status").days_past_due.transform("median"))
    log.add("bureau_credit", "days_past_due", "missing on Active / Written-Off lines", other_missing.sum(), total,
            "median of the same status, flagged")

    opened_late = b.date_opened > AS_OF_DATE
    closed_late = b.date_closed > AS_OF_DATE
    b["date_after_as_of"] = (opened_late | closed_late).astype(int)
    log.add("bureau_credit", "date_opened", f"after the {AS_OF_DATE.date()} as-of date", opened_late.sum(), total,
            "flagged; bureau dates are not used in features")
    log.add("bureau_credit", "date_closed", f"after the as-of date (latest: {b.date_closed.max().date()})",
            closed_late.sum(), total, "flagged; bureau dates are not used in features")

    first_loan = loans.groupby("customer_id").origination_date.min()
    after_first_loan = b.date_opened > b.customer_id.map(first_loan)
    log.add("bureau_credit", "date_opened", "opened after the customer's first loan (not a point-in-time pull)",
            after_first_loan.sum(), total,
            "kept as a current bureau snapshot; production needs bureau data as of each application date")
    return b


def aggregate_bureau(b):
    b = b.assign(
        is_active=b.account_status.eq("Active"),
        is_closed=b.account_status.eq("Closed"),
        is_written_off=b.account_status.eq("Written-Off"),
        is_non_bank=b.lender_category.ne("Bank"),
        is_dpd_30_plus=b.days_past_due.ge(30),
        active_outstanding=b.outstanding_amount.where(b.account_status.eq("Active"), 0),
        active_sanctioned=b.sanctioned_amount.where(b.account_status.eq("Active"), 0),
    )
    g = b.groupby("customer_id")
    agg = pd.DataFrame({
        "bureau_n_lines": g.size(),
        "bureau_n_active": g.is_active.sum(),
        "bureau_n_closed": g.is_closed.sum(),
        "bureau_n_written_off": g.is_written_off.sum(),
        "bureau_n_dpd_30_plus": g.is_dpd_30_plus.sum(),
        "bureau_total_sanctioned": g.sanctioned_amount.sum(),
        "bureau_total_outstanding": g.outstanding_amount.sum(),
        "bureau_max_dpd": g.days_past_due.max(),
        "bureau_mean_dpd": g.days_past_due.mean(),
        "bureau_share_non_bank": g.is_non_bank.mean(),
    })
    active_sanctioned = g.active_sanctioned.sum()
    agg["bureau_active_utilization"] = (g.active_outstanding.sum() / active_sanctioned).where(active_sanctioned > 0)
    agg["bureau_any_written_off"] = (agg.bureau_n_written_off > 0).astype(int)
    agg["bureau_written_off_share"] = agg.bureau_n_written_off / agg.bureau_n_lines
    return agg


# ---------------------------------------------------------------- repayment

def clean_repayment(repayment, loans, log):
    r = repayment.sort_values(["account_id", "payment_month"]).reset_index(drop=True)
    total = len(r)
    n_accounts = r.account_id.nunique()

    months = r.groupby("account_id").payment_month.agg(["size", "min", "max"])
    incomplete = ~(months["size"].eq(12) & months["min"].eq(1) & months["max"].eq(12))
    log.add("repayment_history", "payment_month", "account without exactly months 1-12", incomplete.sum(),
            n_accounts, "check passed")

    paid_in_full = np.isclose(r.emi_paid, r.emi_due)
    expected = np.select(
        [r.days_past_due.eq(0) & paid_in_full,
         r.days_past_due.between(1, 29) & paid_in_full,
         r.days_past_due.ge(30) & r.emi_paid.eq(0)],
        ["On-Time", "Late", "Missed"], default="unexpected")
    log.add("repayment_history", "payment_status", "disagrees with days_past_due / emi_paid",
            (expected != r.payment_status).sum(), total, "check passed")
    log.add("repayment_history", "emi_due", "changes within an account",
            (r.groupby("account_id").emi_due.nunique() > 1).sum(), n_accounts, "check passed")
    log.add("repayment_history", "days_past_due", "90+ days past due inside the 12-month window",
            r.days_past_due.ge(90).sum(), total,
            "check passed: default is not visible in the window, so repayment features do not leak the target")

    # does the reported balance follow each loan's own payment schedule?
    m12 = r[r.payment_month.eq(12)].merge(
        loans[["account_id", "loan_amount", "tenure_months", "interest_rate_pct", "emi"]], on="account_id")
    ever_missed = r.payment_status.eq("Missed").groupby(r.account_id).any()
    m12 = m12[~m12.account_id.map(ever_missed)]
    i = m12.interest_rate_pct / 1200
    growth = (1 + i) ** 12
    scheduled = (m12.loan_amount * growth - m12.emi * (growth - 1) / i).clip(lower=0)
    off_schedule = (m12.outstanding_balance - scheduled).abs() > 0.05 * m12.loan_amount
    log.add("repayment_history", "outstanding_balance",
            "month-12 balance is more than 5% of the loan away from the loan's own payment schedule "
            "(accounts that never missed a payment; e.g. 12-month loans still show ~38% outstanding)",
            off_schedule.sum(), len(m12), "not used as exposure: exposure = original loan_amount")
    return r


def longest_run(flags):
    """Length of the longest streak of consecutive True values."""
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def aggregate_repayment(r):
    r = r.assign(
        on_time=r.payment_status.eq("On-Time"),
        late=r.payment_status.eq("Late"),
        missed=r.payment_status.eq("Missed"),
    )
    g = r.groupby("account_id")
    first3 = r[r.payment_month.le(3)].groupby("account_id")
    last3 = r[r.payment_month.ge(10)].groupby("account_id")
    agg = pd.DataFrame({
        "rep_n_on_time": g.on_time.sum(),
        "rep_n_late": g.late.sum(),
        "rep_n_missed": g.missed.sum(),
        "rep_max_dpd": g.days_past_due.max(),
        "rep_mean_dpd": g.days_past_due.mean(),
        "rep_mean_dpd_first3": first3.days_past_due.mean(),
        "rep_mean_dpd_last3": last3.days_past_due.mean(),
        "rep_n_missed_last3": last3.missed.sum(),
        "rep_paid_ratio": g.emi_paid.sum() / g.emi_due.sum(),
        "rep_max_missed_streak": g.missed.apply(lambda s: longest_run(s.to_numpy())),
        "rep_balance_m12": r[r.payment_month.eq(12)].set_index("account_id").outstanding_balance,
    })
    agg["rep_dpd_trend"] = agg.rep_mean_dpd_last3 - agg.rep_mean_dpd_first3
    agg["rep_ever_30_dpd"] = agg.rep_max_dpd.ge(30).astype(int)
    agg["rep_ever_60_dpd"] = agg.rep_max_dpd.ge(60).astype(int)
    last_missed = r[r.missed].groupby("account_id").payment_month.max()
    # 12 = no missed payment anywhere in the window
    agg["rep_months_since_missed"] = (12 - last_missed).reindex(agg.index).fillna(12).astype(int)
    return agg


# ---------------------------------------------------------------- collections

def clean_collections(collections, loans, log):
    c = collections.merge(loans[["account_id", "origination_date", "loan_amount", "target_default"]],
                          on="account_id", how="left", validate="m:1")
    total = len(c)

    before_loan = c.event_date < c.origination_date
    c["event_before_origination"] = before_loan.astype(int)
    log.add("collections_events", "event_date", "dated before the loan was originated", before_loan.sum(), total,
            "excluded from collections metrics, flagged")

    paid_nothing = c.outcome.eq("Payment Received") & c.amount_recovered.eq(0)
    silent_payer = c.outcome.eq("No Response") & c.amount_recovered.gt(0)
    c["outcome_amount_conflict"] = (paid_nothing | silent_payer).astype(int)
    log.add("collections_events", "outcome", "'Payment Received' but 0 recovered", paid_nothing.sum(), total,
            "flagged")
    log.add("collections_events", "outcome", "'No Response' but money recovered", silent_payer.sum(), total,
            "flagged")
    log.add("collections_events", "action_type", "'Written-Off Referral' on a loan marked as performing",
            (c.action_type.eq("Written-Off Referral") & c.target_default.eq(0)).sum(), total, "flagged")
    over_recovered = c.amount_recovered > c.loan_amount
    c["recovered_above_loan"] = over_recovered.astype(int)
    log.add("collections_events", "amount_recovered", "more than the original loan amount", over_recovered.sum(),
            total, "flagged")
    log.add("collections_events", "(whole table)",
            "dates unreliable and no predictive lift beyond repayment history (checked in the audit)",
            total, total, "shown in the dashboard only; left out of the model")
    # target-derived columns must not travel downstream with collections data
    return c.drop(columns=["origination_date", "loan_amount", "target_default"])


def aggregate_collections(c):
    valid = c[c.event_before_origination.eq(0)].sort_values("event_date")
    g = valid.groupby("account_id")
    valid_agg = pd.DataFrame({
        "coll_n_events": g.size(),
        "coll_total_recovered": g.amount_recovered.sum(),
        "coll_last_action": g.action_type.last(),
        "coll_last_outcome": g.outcome.last(),
        "coll_last_event_date": g.event_date.max(),
    })
    return c.groupby("account_id").size().rename("coll_n_events_raw").to_frame().join(valid_agg)


# ---------------------------------------------------------------- join

def build_accounts(loans, customers, bureau_agg, repayment_agg, collections_agg, log):
    df = (loans
          .merge(customers, on="customer_id", how="left", validate="m:1")
          .merge(bureau_agg, left_on="customer_id", right_index=True, how="left", validate="m:1")
          .merge(repayment_agg, left_on="account_id", right_index=True, how="left", validate="1:1")
          .merge(collections_agg, left_on="account_id", right_index=True, how="left", validate="1:1"))
    assert len(df) == len(loans) and df.account_id.is_unique, "join changed the number of loan rows"

    df["bureau_no_file"] = df.bureau_n_lines.isna().astype(int)
    count_cols = [col for col in bureau_agg.columns if col.startswith(("bureau_n_", "bureau_total_", "bureau_any_"))]
    df[count_cols] = df[count_cols].fillna(0)
    df[["coll_n_events", "coll_n_events_raw", "coll_total_recovered"]] = (
        df[["coll_n_events", "coll_n_events_raw", "coll_total_recovered"]].fillna(0))

    df["exposure"] = df.loan_amount
    df["emi_to_income"] = df.emi / (df.annual_income / 12)
    df["emi_exceeds_income"] = df.emi_to_income.gt(1).astype(int)
    log.add("loan_accounts", "emi", "monthly payment is larger than the borrower's whole monthly income",
            df.emi_exceeds_income.sum(), len(df),
            "kept (the exposure is real), flagged; would fail any affordability check at origination")
    df["emi_to_income_band"] = pd.cut(df.emi_to_income, [0, 0.1, 0.2, 0.3, 0.5, np.inf], right=False,
                                      labels=["<10%", "10-20%", "20-30%", "30-50%", "50%+"]).astype(str)
    df["loan_to_income"] = df.loan_amount / df.annual_income
    df["lti_band"] = pd.cut(df.loan_to_income, [0, 0.25, 0.5, 1, 2, np.inf], right=False,
                            labels=["<0.25x", "0.25-0.5x", "0.5-1x", "1-2x", "2x+"]).astype(str)
    df["bureau_debt_to_income"] = df.bureau_total_outstanding / df.annual_income
    df["n_loans_customer"] = df.groupby("customer_id").account_id.transform("size")
    df["customer_total_exposure"] = df.groupby("customer_id").loan_amount.transform("sum")

    first = ["account_id", "customer_id", "target_default", "loan_type", "loan_purpose", "loan_amount", "exposure",
             "interest_rate_pct", "tenure_months", "emi", "origination_date", "origination_year",
             "origination_quarter", "disbursal_channel", "collateral_flag"]
    return df[first + [col for col in df.columns if col not in first]]


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    log = DQLog()

    customers, loans, bureau, repayment, collections = load_raw()
    check_keys(customers, loans, bureau, repayment, collections, log)

    customers_c = clean_customers(customers, loans, log)
    loans_c = clean_loans(loans, repayment, log)
    bureau_c = clean_bureau(bureau, customers_c, loans_c, log)
    repayment_c = clean_repayment(repayment, loans_c, log)
    collections_c = clean_collections(collections, loans_c, log)

    accounts = build_accounts(loans_c, customers_c, aggregate_bureau(bureau_c), aggregate_repayment(repayment_c),
                              aggregate_collections(collections_c), log)

    accounts.to_csv(PROCESSED / "accounts_clean.csv", index=False)
    customers_c.to_csv(PROCESSED / "customers_clean.csv", index=False)
    bureau_c.to_csv(PROCESSED / "bureau_clean.csv", index=False)
    repayment_c.to_csv(PROCESSED / "repayment_clean.csv", index=False)
    collections_c.to_csv(PROCESSED / "collections_clean.csv", index=False)
    dq = log.to_frame()
    dq.to_csv(PROCESSED / "dq_log.csv", index=False)

    print(f"accounts_clean.csv: {accounts.shape[0]:,} rows x {accounts.shape[1]} columns, "
          f"default rate {accounts.target_default.mean():.1%}")
    print(f"dq_log.csv: {len(dq)} checks, {(dq.action != 'check passed').sum()} with issues\n")
    with pd.option_context("display.max_colwidth", 70, "display.width", 250):
        print(dq[["table", "column", "rows_affected", "pct_of_rows", "issue"]].to_string(index=False))


if __name__ == "__main__":
    main()
