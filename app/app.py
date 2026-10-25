"""
Credit Risk Portfolio dashboard.

    python run_pipeline.py        (once, builds the data the dashboard reads)
    streamlit run app/app.py

Pages follow the assignment brief and share the sidebar filters:
  Dashboard     Portfolio view, Segment drill-down, Account-level view
  AI feature    Ask the Portfolio (grounded natural-language Q&A)
  Model         performance and monitoring
  Bonus AI      Anomaly review (a second AI feature, beyond the brief)
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path[:0] = [str(ROOT / "src"), str(APP)]

import charts as ch  # noqa: E402
import chat  # noqa: E402
import portfolio as pf  # noqa: E402
from config import PROCESSED  # noqa: E402

REPORTS = ROOT / "reports"
REVIEW_DIR = ROOT / "data" / "review"
REVIEW_LOG = REVIEW_DIR / "anomaly_reviews.csv"
FEEDBACK_LOG = REVIEW_DIR / "chat_feedback.csv"
REQUIRED = [PROCESSED / "accounts_clean.csv", PROCESSED / "scores.csv", PROCESSED / "anomalies.csv",
            REPORTS / "model_metrics.json", REPORTS / "anomaly_eval.json"]

st.set_page_config(page_title="Credit Risk Portfolio", page_icon=":material/account_balance:", layout="wide")
st.markdown("""<style>
  .block-container {padding-top: 2rem; max-width: 1400px;}
  [data-testid="stMetricValue"] {font-size: 1.7rem;}
  .pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.85rem; font-weight:600;
         border:1px solid rgba(11,11,11,0.10); margin-right:6px;}
  .muted {color:#52514e; font-size:0.9rem;}
</style>""", unsafe_allow_html=True)

if missing := [p.name for p in REQUIRED if not p.exists()]:
    st.error(f"The dashboard data hasn't been built yet (missing: {', '.join(missing)}). Run this once:")
    st.code("python run_pipeline.py")
    st.stop()


@st.cache_data(show_spinner="Loading portfolio...")
def load():
    book = pf.load_book()
    book["exposure"] = book.exposure.round().astype(int)  # whole dollars in every table
    repayment = pd.read_csv(PROCESSED / "repayment_clean.csv")
    bureau = pd.read_csv(PROCESSED / "bureau_clean.csv", parse_dates=["date_opened", "date_closed"])
    collections = pd.read_csv(PROCESSED / "collections_clean.csv", parse_dates=["event_date"])
    reports = {name: pd.read_csv(REPORTS / f"{name}.csv") for name in
               ["test_predictions", "deciles_test", "risk_bands_test", "feature_importance", "fairness",
                "csi_baseline"]}
    metrics = json.loads((REPORTS / "model_metrics.json").read_text())
    anomaly_eval = json.loads((REPORTS / "anomaly_eval.json").read_text())
    return book, repayment, bureau, collections, reports, metrics, anomaly_eval


book, repayment, bureau, collections, reports, metrics, anomaly_eval = load()
BANDS = pf.ORDER["risk_band"]


# ---------------------------------------------------------------- helpers

def pct(x, digits=1):
    return "n/a" if pd.isna(x) else f"{x:.{digits}%}"


def options(column):
    values = book[column].dropna().astype(str).unique().tolist()
    order = pf.ORDER.get(column)
    return [v for v in order if v in values] if order else sorted(values)


def band_pill(band):
    color = ch.STATUS.get(band, "#898781")
    ink = "#0b0b0b" if band == "Medium" else "#ffffff"
    return f'<span class="pill" style="background:{color};color:{ink}">{band} risk</span>'


def severity_pill(severity):
    if not severity:
        return '<span class="pill" style="background:#f0efec;color:#52514e">Not flagged</span>'
    color = ch.STATUS["Very High"] if severity == "High" else ch.STATUS["Medium"]
    ink = "#ffffff" if severity == "High" else "#0b0b0b"
    return f'<span class="pill" style="background:{color};color:{ink}">Anomaly: {severity}</span>'


def md(text):
    """Markdown with dollar signs escaped (Streamlit would read $...$ as maths)."""
    return text.replace("$", "\\$")


def open_account(account_id):
    st.session_state["account_id"] = account_id
    st.switch_page(PAGES["account"])


# shared column labels and formats for account tables
ACCOUNT_COLUMNS = {
    "account_id": "Account", "loan_type": "Loan type", "region": "Region", "risk_band": "Risk band",
    "exposure": st.column_config.NumberColumn("Exposure ($)", format="localized"),
    "pd": st.column_config.ProgressColumn("Predicted PD", format="%.1f%%", min_value=0, max_value=100),
    "anomaly_severity": "Anomaly", "anomaly_trigger": "Trigger",
    "anomaly_reasons": st.column_config.TextColumn("Why it was flagged", width="large"),
    "top_risk_drivers": st.column_config.TextColumn("Main risk drivers", width="large"),
    "target_default": st.column_config.CheckboxColumn("Defaulted"), "status": "Status",
}


def account_table(table, column_config=None, **kwargs):
    """An account table with PD shown as a percentage to one decimal; returns the selection event."""
    shown = table.assign(pd=table.pd * 100) if "pd" in table else table
    return st.dataframe(shown, hide_index=True, use_container_width=True,
                        column_config=ACCOUNT_COLUMNS | (column_config or {}), **kwargs)


def read_log(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def append_log(path, row):
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


# ---------------------------------------------------------------- sidebar filters (shared by every page)

FILTERS = {"region": "Region", "loan_type": "Loan type", "employment_type": "Employment type",
           "income_band": "Income band", "age_band": "Age band", "origination_year": "Origination year",
           "risk_band": "Model risk band"}


def clear_filters():
    for column in FILTERS:
        st.session_state[f"filter_{column}"] = []


with st.sidebar:
    st.markdown("### Filters")
    st.caption("Apply to every page. Empty = all.")
    selected = {column: st.multiselect(label, options(column), key=f"filter_{column}", placeholder="All")
                for column, label in FILTERS.items()}
    st.button("Clear filters", on_click=clear_filters, use_container_width=True)
    view = pf.apply_filters(book, selected)
    filtered = len(view) < len(book)
    st.caption(f"Showing **{len(view):,}** of {len(book):,} accounts")

if view.empty:
    st.warning("No accounts match these filters. Clear some filters in the sidebar.")
    st.stop()


def kpi_row(df):
    o, full = pf.overview(df), pf.overview(book)
    delta = (lambda key: f"{(o[key] - full[key]) * 100:+.1f} pts") if filtered else (lambda key: None)
    cols = st.columns(6)
    cols[0].metric("Accounts", f"{o['accounts']:,}", help=f"{o['customers']:,} customers")
    cols[1].metric("Exposure", ch.money(o["exposure"]), help="Original loan amount")
    cols[2].metric("Default rate", pct(o["default_rate"]), delta("default_rate"), delta_color="inverse",
                   help="Share of accounts that defaulted (observed). Delta: vs whole portfolio")
    cols[3].metric("Avg PD", pct(o["avg_predicted_pd"]), delta("avg_predicted_pd"), delta_color="inverse",
                   help="Average predicted probability of default. Delta: vs whole portfolio")
    cols[4].metric("Exp. loss", ch.money(o["expected_loss"]),
                   help="Expected loss = PD x illustrative loss-given-default x exposure (assumptions log)")
    cols[5].metric("Anomalies", f"{o['flagged_anomalies']:,}",
                   help="Accounts in the anomaly review queue (see Anomaly review)")


# ---------------------------------------------------------------- page: portfolio overview

def page_portfolio():
    st.title("Portfolio view")
    st.caption("Where the risk and the exposure sit. Exposure = original loan amount; default rate = observed outcome.")
    kpi_row(view)

    dims = ["loan_type", "income_band", "employment_type", "region", "age_band", "emi_to_income_band", "lti_band"]
    dim = st.segmented_control("Break down by", dims, format_func=lambda d: pf.DIMENSIONS[d],
                               default="loan_type", key="overview_dim") or "loan_type"
    table = pf.segment_summary(view, dim)
    if dim not in pf.ORDER:
        table = table.sort_values("default_rate", ascending=False)
    average = view.target_default.mean()
    worst = table.loc[table.default_rate.idxmax()]
    st.markdown(md(f"**{pf.DIMENSIONS[dim]}:** the highest default rate is **{worst.segment}** at "
                   f"**{pct(worst.default_rate)}** ({worst.lift:.2f}x the average of {pct(average)}), holding "
                   f"{pct(worst.exposure_share)} of exposure and {pct(worst.share_of_defaulted_exposure)} of "
                   f"defaulted exposure."))
    if table.small_sample.any():
        st.caption("Segments with fewer than 100 accounts: " + ", ".join(table[table.small_sample].segment)
                   + ". Treat their rates with caution.")

    left, right = st.columns(2)
    with left:
        st.markdown(f"##### Default rate by {pf.DIMENSIONS[dim].lower()}")
        st.plotly_chart(ch.rate_bars(table, average=average, hover=["accounts"]), use_container_width=True,
                        config=ch.CONFIG)
    with right:
        st.markdown("##### Where the money is vs where the losses are")
        st.plotly_chart(ch.share_bars(table), use_container_width=True, config=ch.CONFIG)

    left, right = st.columns(2)
    bands = pf.segment_summary(view, "risk_band")
    with left:
        st.markdown("##### Model risk bands: share of exposure")
        st.plotly_chart(ch.band_bars(bands, "exposure_share"), use_container_width=True, config=ch.CONFIG)
        st.caption("Observed default rate per band: " + " · ".join(
            f"{r.segment} {pct(r.default_rate)}" for r in bands.itertuples()))
    with right:
        st.markdown("##### Default rate by loan origination period")
        by = st.segmented_control("Trend by", ["origination_year", "origination_quarter"],
                                  format_func=lambda x: x.split("_")[1].title(), default="origination_year",
                                  key="trend_by", label_visibility="collapsed") or "origination_year"
        trend = pf.vintage_trend(view, by)
        st.plotly_chart(ch.trend_lines(trend, by, {"default_rate": "Observed default rate",
                                                   "avg_predicted_pd": "Average predicted PD"}),
                        use_container_width=True, config=ch.CONFIG)

    with st.expander("Table view"):
        st.dataframe(table.drop(columns=["defaulted_exposure"]), hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="percent") for c in
                                    ["default_rate", "account_share", "exposure_share", "share_of_defaulted_exposure",
                                     "avg_predicted_pd"]} | {"exposure": st.column_config.NumberColumn(format="dollar")})


# ---------------------------------------------------------------- page: segment explorer

def page_segments():
    st.title("Segment drill-down")
    st.caption("Cross two dimensions, then drill into any cell down to individual accounts.")
    dims = [d for d in pf.DIMENSIONS if d != "origination_year"] + ["origination_year"]
    c1, c2 = st.columns(2)
    row = c1.selectbox("Rows", dims, index=dims.index("income_band"), format_func=pf.DIMENSIONS.get)
    col = c2.selectbox("Columns", [d for d in dims if d != row], format_func=pf.DIMENSIONS.get,
                       index=[d for d in dims if d != row].index("employment_type") if row != "employment_type" else 0)
    rate, count = pf.cross_tab(view, row, col)
    average = view.target_default.mean()
    st.plotly_chart(ch.heatmap(rate, count, average, pf.DIMENSIONS[row], pf.DIMENSIONS[col]),
                    use_container_width=True, config=ch.CONFIG)
    st.caption(f"Colour shows distance from the average of the current view ({pct(average)}): red above, blue below. "
               "\\* = fewer than 100 accounts, treat with caution.")

    st.markdown("#### Drill down")
    st.caption("Opens on the riskiest cell with enough accounts to trust; pick any other cell below.")
    rows_, cols_ = [str(v) for v in rate.index], [str(v) for v in rate.columns]
    cells = rate.stack().rename("rate").to_frame().join(count.stack().rename("n"))
    trusted = cells[cells.n >= pf.SMALL_SEGMENT]
    pool = trusted if len(trusted) else cells[cells.n > 0]
    start_row, start_col = (str(v) for v in pool.rate.idxmax())
    d1, d2 = st.columns(2)
    r_value = d1.selectbox(pf.DIMENSIONS[row], rows_, index=rows_.index(start_row))
    c_value = d2.selectbox(pf.DIMENSIONS[col], cols_, index=cols_.index(start_col))
    segment = view[view[row].astype(str).eq(r_value) & view[col].astype(str).eq(c_value)]
    if segment.empty:
        st.info("No accounts in this cell.")
        return
    o = pf.overview(segment)
    k = st.columns(5)
    k[0].metric("Accounts", f"{o['accounts']:,}")
    k[1].metric("Default rate", pct(o["default_rate"]), f"{(o['default_rate'] - average) * 100:+.1f} pts",
                delta_color="inverse", help="Delta: vs the average of the current view")
    k[2].metric("Exposure", ch.money(o["exposure"]))
    k[3].metric("Avg predicted PD", pct(o["avg_predicted_pd"]))
    k[4].metric("Flagged anomalies", f"{o['flagged_anomalies']:,}")
    st.caption("Accounts in this cell, riskiest first. Select a row to open the account.")
    table = segment.sort_values("pd", ascending=False)[
        ["account_id", "loan_type", "region", "exposure", "pd", "risk_band", "anomaly_severity", "top_risk_drivers",
         "target_default"]]
    event = account_table(table, on_select="rerun", selection_mode="single-row", key="segment_table")
    if event.selection.rows:
        open_account(table.iloc[event.selection.rows[0]].account_id)


# ---------------------------------------------------------------- page: account view

def page_account():
    st.title("Account-level view")
    ranked = view.sort_values("pd", ascending=False).account_id.tolist()
    current = st.session_state.get("account_id")
    if current not in ranked:
        current = ranked[0]
    account_id = st.selectbox("Account (riskiest first, type to search)", ranked, index=ranked.index(current))
    st.session_state["account_id"] = account_id
    a = book[book.account_id.eq(account_id)].iloc[0]

    st.markdown(f"{band_pill(a.risk_band)}{severity_pill(a.anomaly_severity)}"
                f'<span class="muted">{a.loan_type} · {a.region} · originated {a.origination_date:%d %b %Y} · '
                f'customer {a.customer_id}</span>', unsafe_allow_html=True)
    k = st.columns(5)
    k[0].metric("Predicted PD", pct(a.pd), help="Champion model probability of default")
    k[1].metric("Riskier than", f"{int(a.pd_percentile * 100)}% of accounts",
                help="Share of the portfolio with a lower predicted PD")
    k[2].metric("Exposure", ch.money(a.exposure))
    k[3].metric("Expected loss", ch.money(a.expected_loss), help=f"Illustrative LGD {a.lgd_assumed:.0%}")
    k[4].metric("Observed outcome", "Defaulted" if a.target_default else "Performing")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("##### Why this score: key drivers")
        st.plotly_chart(ch.driver_bars(json.loads(a.drivers_json)), use_container_width=True, config=ch.CONFIG)
        st.caption("Each bar is how far that factor pushes this account's risk above (red) or below (blue) "
                   "an average account. These are exact for the logistic regression champion.")
    with right:
        st.markdown("##### Repayment history (first 12 months)")
        history = repayment[repayment.account_id.eq(account_id)]
        st.plotly_chart(ch.repayment_bars(history), use_container_width=True, config=ch.CONFIG)
        st.caption(f"{a.rep_n_on_time:.0f} on time · {a.rep_n_late:.0f} late · {a.rep_n_missed:.0f} missed · "
                   f"worst {a.rep_max_dpd:.0f} days past due · {a.rep_paid_ratio:.0%} of instalments paid")
        if a.anomaly_flag:
            st.markdown("##### Anomaly flags")
            for reason in a.anomaly_reasons.split("; "):
                st.markdown(md(f"- {reason}"))

    loan_tab, customer_tab, bureau_tab, other_tab, collections_tab = st.tabs(
        ["Loan", "Customer", "Bureau lines", "Other loans of this customer", "Collections"])
    with loan_tab:
        st.dataframe(pd.DataFrame({
            "Loan type": a.loan_type, "Purpose (unreliable)": a.loan_purpose, "Amount": ch.money(a.loan_amount),
            "Interest rate": f"{a.interest_rate_pct:.2f}%" + (" (derived)" if a.interest_rate_derived else ""),
            "Tenure": f"{a.tenure_months:.0f} months", "Monthly instalment": f"${a.emi:,.0f}",
            "Instalment / income": f"{a.emi_to_income:.0%}", "Loan / income": f"{a.loan_to_income:.2f}x",
            "Channel": a.disbursal_channel}, index=["value"]).T, use_container_width=True)
    with customer_tab:
        st.dataframe(pd.DataFrame({
            "Age": a.age, "Employment": a.employment_type, "Sector": a.occupation_sector,
            "Years in job": a.years_employed, "Annual income": ch.money(a.annual_income), "Education": a.education_level,
            "Housing": a.housing_type, "City tier": a.city_tier, "Dependents": a.num_dependents},
            index=["value"]).T.astype(str), use_container_width=True)
    with bureau_tab:
        lines = bureau[bureau.customer_id.eq(a.customer_id)]
        if lines.empty:
            st.info("No bureau record (thin file).")
        else:
            st.dataframe(lines[["credit_type", "lender_category", "account_status", "sanctioned_amount",
                                "outstanding_amount", "days_past_due", "dpd_imputed"]],
                         hide_index=True, use_container_width=True)
    with other_tab:
        others = book[book.customer_id.eq(a.customer_id) & book.account_id.ne(account_id)]
        if others.empty:
            st.info("This customer has no other loans with the bank.")
        else:
            account_table(others[["account_id", "loan_type", "exposure", "pd", "risk_band", "target_default"]])
    with collections_tab:
        events = collections[collections.account_id.eq(account_id)]
        if events.empty:
            st.info("No collections activity.")
        else:
            st.caption("Events dated before the loan existed are data errors and excluded from metrics.")
            st.dataframe(events[["event_date", "action_type", "outcome", "amount_recovered",
                                 "event_before_origination"]], hide_index=True, use_container_width=True)


# ---------------------------------------------------------------- page: anomaly review (bonus AI feature)

def page_anomalies():
    st.title("Bonus AI: anomaly review")
    st.caption("A second AI feature, beyond the one the brief asks for: accounts whose repayment, bureau or "
               "affordability pattern is unusual for their peer group or their own history. An analyst confirms or "
               "dismisses every flag; nothing happens to a customer automatically.")
    with st.expander("How the detector works: triggers, false alarms, human review"):
        st.markdown(md(
            "**What triggers a flag** (refreshed with each monthly data load)\n"
            "- **Peer outlier**: an Isolation Forest scores how unusual each account's mix of repayment, bureau and "
            "affordability metrics is *within its peer group* (same loan type and income band). The 5% most "
            "unusual are flagged; the top 1% are High severity.\n"
            "- **Affordability breach**: the monthly instalment is larger than the borrower's monthly income (High).\n"
            "- **Payment shock**: 2+ missed payments in the last 3 months after 9 months with no missed payment, "
            "compared with the account's own history.\n\n"
            "**False positive**: a flag on an account that is fine, e.g. a large loan for a customer whose income "
            "isn't fully recorded, or a one-off missed payment caused by a bank holiday. About two thirds of flagged "
            "accounts don't default, so a flag is a prompt to look, not a verdict.\n\n"
            "**False negative**: a defaulter whose behaviour looks normal for its peers. The detector catches about 1 "
            "in 9 defaults; the PD model covers the rest of the book.\n\n"
            "**Human in the loop**: an analyst reviews the queue (High first), confirms (escalate to early-intervention "
            "contact) or dismisses with a reason. Decisions are logged with name and time; dismissal reasons are "
            "reviewed monthly to tune thresholds."))

    reviews = read_log(REVIEW_LOG)
    latest = (reviews.sort_values("timestamp").groupby("account_id").tail(1).set_index("account_id")
              if not reviews.empty else pd.DataFrame(columns=["decision"]))
    queue = view[view.anomaly_flag.eq(1)].copy()
    queue["status"] = queue.account_id.map(latest["decision"]).fillna("To review") if not latest.empty else "To review"

    e = anomaly_eval
    view_rate = view.target_default.mean()
    k = st.columns(5)
    k[0].metric("Flagged in view", f"{len(queue):,}", help=f"{e['flagged']:,} across the whole portfolio")
    k[1].metric("High severity", f"{queue.anomaly_severity.eq('High').sum():,}")
    k[2].metric("Default rate of flagged", pct(queue.target_default.mean()) if len(queue) else "n/a",
                f"{queue.target_default.mean() / view_rate:.1f}x the view" if len(queue) and view_rate else None,
                delta_color="off")
    k[3].metric("Still to review", f"{queue.status.eq('To review').sum():,}")
    k[4].metric("Also High risk in model",
                pct(queue.risk_band.isin(["High", "Very High"]).mean(), 0) if len(queue) else "n/a",
                help="Share of the flags in view that the PD model also rates High or Very High. The rest are flags "
                     "the PD model alone would not prioritise.")

    f1, f2, f3 = st.columns([1, 1.3, 1.3])
    severity = f1.multiselect("Severity", ["High", "Medium"], placeholder="All")
    trigger = f2.multiselect("Trigger", ["Peer outlier", "Affordability breach", "Payment shock"], placeholder="All")
    status = f3.segmented_control("Status", ["To review", "Confirmed", "Dismissed", "All"], default="To review") or "All"
    shown = queue
    if severity:
        shown = shown[shown.anomaly_severity.isin(severity)]
    if trigger:
        shown = shown[shown.anomaly_trigger.apply(lambda t: any(x in t for x in trigger))]
    if status != "All":
        shown = shown[shown.status.eq(status)]
    shown = shown.assign(severity_rank=shown.anomaly_severity.map({"High": 0, "Medium": 1})).sort_values(
        ["severity_rank", "anomaly_score"], ascending=[True, False])
    table = shown[["account_id", "anomaly_severity", "anomaly_trigger", "anomaly_reasons", "pd", "risk_band",
                   "exposure", "status"]]
    event = account_table(table, column_config={"anomaly_severity": "Severity"}, on_select="rerun",
                          selection_mode="single-row", key="anomaly_table", height=320)
    if not event.selection.rows:
        st.caption("Select a row to review it.")
    else:
        a = table.iloc[event.selection.rows[0]]
        st.markdown(f"#### Review {a.account_id}")
        st.markdown(f"{severity_pill(a.anomaly_severity)}{band_pill(a.risk_band)}"
                    f'<span class="muted">predicted PD {pct(a.pd)} · exposure {ch.money(a.exposure)}</span>',
                    unsafe_allow_html=True)
        for reason in a.anomaly_reasons.split("; "):
            st.markdown(md(f"- {reason}"))
        if st.button("Open full account view", key="open_from_queue"):
            open_account(a.account_id)
        with st.form("review_form", clear_on_submit=True):
            decision = st.radio("Decision", ["Confirmed", "Dismissed"], horizontal=True,
                                captions=["Real concern: escalate to early-intervention contact",
                                          "Not a concern: record why"])
            reason = st.selectbox("Reason (for dismissals)", ["", "Data error", "Known customer circumstance",
                                                               "Benign pattern for this customer", "Other"])
            analyst = st.text_input("Analyst name")
            note = st.text_area("Note", height=80)
            if st.form_submit_button("Save decision", type="primary"):
                if not analyst.strip():
                    st.error("Add your name so the decision is traceable.")
                elif decision == "Dismissed" and not reason:
                    st.error("Pick a reason for the dismissal.")
                else:
                    append_log(REVIEW_LOG, {"timestamp": datetime.now().isoformat(timespec="seconds"),
                                            "account_id": a.account_id, "decision": decision, "reason": reason,
                                            "analyst": analyst.strip(), "note": note.strip(),
                                            "severity": a.anomaly_severity, "trigger": a.anomaly_trigger})
                    st.success(f"Saved: {a.account_id} {decision.lower()}.")
                    st.rerun()

    st.markdown("##### Does the detector concentrate risk? Default rate by trigger (whole portfolio)")
    triggers = pd.DataFrame([{"segment": name, "default_rate": v["default_rate"], "accounts": v["accounts"]}
                             for name, v in e["by_trigger"].items()])
    st.plotly_chart(ch.rate_bars(triggers, average=e["portfolio_default_rate"], hover=["accounts"], height=220),
                    use_container_width=True, config=ch.CONFIG)
    if not reviews.empty:
        with st.expander(f"Review log ({len(reviews)} decisions)"):
            st.dataframe(reviews.sort_values("timestamp", ascending=False), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------- page: ask the portfolio (the AI feature)

def show_check(v):
    """The grounding check under an answer: what matched the data, and what the analyst must look at."""
    if v["status"] == "verified":
        st.caption(f":material/verified: All {v['numbers_checked']} numbers were found in the data the tools returned, "
                   "and none sits next to a segment it doesn't belong to.")
        return
    if v["status"] != "check":
        return
    missing = [n for n in v["unverified"] if n not in v["from_question"]]
    problems = [f"{', '.join(missing)} not found in the data"] if missing else []
    problems += [f"{m['number']} is in the data for {' / '.join(m['belongs_to'])}, not for "
                 f"{' / '.join(m['named_in_sentence'])}" for m in v["misattributed"]]
    if problems:
        st.warning(md("Check before using: " + "; ".join(problems) + "."), icon=":material/warning:")
    if v["from_question"]:
        st.caption(md(f":material/info: {', '.join(v['from_question'])} repeats your question; it is not a figure "
                      "from the data."))


def show_answer(result, index):
    st.markdown(md(result["answer"]))
    v = result["verification"]
    if result["mode"] == "offline":
        st.caption(f":material/cloud_off: Offline mode, answered by fixed rules ({result['reason']}).")
    show_check(v)
    with st.expander(f"Data behind this answer ({len(result['tool_calls'])} tool call"
                     f"{'s' if len(result['tool_calls']) != 1 else ''})"):
        for call in result["tool_calls"]:
            st.markdown(f"`{call['tool']}` {json.dumps(call['arguments'])}")
            res = call["result"]
            key = next((k for k in ["segments", "ranked_segments", "cells_sorted_by_default_rate", "periods",
                                    "accounts"] if k in res),
                       None)
            if key:
                st.dataframe(pd.DataFrame(res[key]), hide_index=True, use_container_width=True)
            else:
                st.json(res, expanded=False)
        if v.get("sources"):
            st.markdown("**Where each number came from**")
            st.dataframe(pd.DataFrame(v["sources"]).rename(columns={"number": "Number in the answer",
                                                                    "source": "Found in"}),
                         hide_index=True, use_container_width=True)
    feedback = st.feedback("thumbs", key=f"feedback_{index}")
    if feedback is not None and not st.session_state.get(f"logged_{index}"):
        append_log(FEEDBACK_LOG, {"timestamp": datetime.now().isoformat(timespec="seconds"),
                                  "question": result.get("question", ""), "answer": result["answer"],
                                  "helpful": bool(feedback), "verification": v["status"],
                                  "model": result.get("model")})
        st.session_state[f"logged_{index}"] = True


def page_chat():
    st.title("AI feature: Ask the Portfolio")
    key = chat.api_key()
    if key:
        st.caption(f":material/smart_toy: Natural-language Q&A over the dashboard data, answered by {chat.MODELS[0]} "
                   f"(NVIDIA) through {len(chat.TOOLS)} query tools. The model never sees raw data or calculates; every "
                   "number in an answer is checked against the data the tools returned, and against the segment it is "
                   "attached to, so an invented or misplaced number is flagged.")
    else:
        st.caption(":material/cloud_off: No NVIDIA API key found, so the chat runs in offline mode: fixed rules "
                   "answer the common questions with the same tools and the same number check. Add NVIDIA_API_KEY to "
                   ".streamlit/secrets.toml to enable the LLM.")
    boxes = [
        ("What triggers it", "A question typed below, or a suggested question. Optionally limited to the sidebar "
                             "filters, so it answers over exactly what the dashboard shows."),
        ("False positive", "A confident answer the data can't really support, e.g. a misread period or filter. "
                           "Each answer shows the exact queries behind it, and any number that is not in the data, "
                           "or sits next to the wrong segment, is flagged."),
        ("False negative", "\"I can't answer that\" when the data could. Rephrase, or use the Portfolio view and "
                           "Segment drill-down for the same cut."),
        ("Human in the loop", "Answers are for exploration. Before a figure goes into a decision or report, the "
                              "analyst checks the data behind it and rates the answer; ratings are logged and "
                              "reviewed monthly."),
    ]
    for column, (heading, text) in zip(st.columns(4), boxes):
        with column.container(border=True):
            st.markdown(f"**{heading}**")
            st.caption(md(text))
    scoped = st.toggle("Answer within the sidebar filters", value=False,
                       help="Off: the whole portfolio. On: only the accounts in the current filtered view.")
    data = view if scoped else book

    if "chat" not in st.session_state:
        st.session_state.chat = []
    cols = st.columns(3)
    for i, question in enumerate(chat.SUGGESTED_QUESTIONS):
        if cols[i % 3].button(question, key=f"suggest_{i}", use_container_width=True):
            st.session_state.pending = question
    if st.session_state.chat and st.button("Clear conversation", icon=":material/delete:"):
        st.session_state.chat = []
        st.rerun()

    for i, turn in enumerate(st.session_state.chat):
        with st.chat_message(turn["role"]):
            if turn["role"] == "user":
                st.markdown(md(turn["content"]))
            else:
                show_answer(turn["result"], i)

    question = st.chat_input("Ask about the portfolio, e.g. 'default rate by region for mortgages'")
    question = question or st.session_state.pop("pending", None)
    if question:
        history = [{"role": t["role"], "content": t["content"]} for t in st.session_state.chat]
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(md(question))
        with st.chat_message("assistant"):
            with st.spinner("Querying the portfolio..."):
                result = chat.ask(data, question, history=history, key=key)
            result["question"] = question
            st.session_state.chat.append({"role": "assistant", "content": result["answer"], "result": result})
            show_answer(result, len(st.session_state.chat) - 1)


# ---------------------------------------------------------------- page: model & monitoring

def page_model():
    st.title("Model performance & monitoring")
    champion = metrics["champion"]
    names = {"logistic": "Logistic regression", "lightgbm": "LightGBM"}
    rows = []
    for kind in ["logistic", "lightgbm"]:
        m = metrics["models"][kind]
        rows.append({"Model": names[kind] + (" (champion)" if kind == champion else " (challenger)"),
                     "CV AUC": f"{m['cv_auc_mean']:.3f} ± {m['cv_auc_std']:.3f}", "Test AUC": round(m["test"]["auc"], 3),
                     "Test Gini": round(m["test"]["gini"], 3), "Test KS": round(m["test"]["ks"], 3),
                     "Brier": round(m["test"]["brier"], 4), "Out-of-time AUC": round(m["out_of_time"]["auc"], 3),
                     "Train AUC": round(m["train"]["auc"], 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    gain = metrics["lightgbm_auc_gain_on_test"]
    st.info(f"**{names[champion]} is the champion.** LightGBM changed test AUC by {gain:+.3f}; it had to improve it "
            f"by at least {metrics['materiality_auc']} to justify a less transparent model. Its train AUC "
            f"({metrics['models']['lightgbm']['train']['auc']:.3f}) vs test AUC also shows it overfits. The "
            "logistic regression's drivers are exact, which supports adverse-action reason codes.",
            icon=":material/emoji_events:")
    split = metrics["split"]
    st.caption(f"Hold-out test: {split['test_accounts']:,} accounts from customers never seen in training. "
               f"Out-of-time: trained on 2018-2021 loans ({split['oot_development_accounts']:,}), tested on "
               f"{split['oot_test_accounts']:,} 2022-2023 loans of new customers.")

    test = reports["test_predictions"]
    left, right = st.columns(2)
    with left:
        st.markdown("##### ROC curve, hold-out test")
        st.plotly_chart(ch.roc_chart(test.target_default, {"Logistic regression": test.pd_logistic,
                                                           "LightGBM": test.pd_lightgbm}),
                        use_container_width=True, config=ch.CONFIG)
    with right:
        st.markdown("##### Calibration: predicted vs actual default rate by risk band (test)")
        bands = reports["risk_bands_test"]
        st.plotly_chart(ch.paired_bars(bands, "risk_band", "avg_predicted_pd", "actual_default_rate",
                                       ["Predicted PD", "Actual default rate"]),
                        use_container_width=True, config=ch.CONFIG)

    left, right = st.columns(2)
    with left:
        deciles = reports["deciles_test"].assign(segment=lambda d: "Decile " + d.decile.astype(str))
        top3 = deciles.cumulative_share_of_defaults.iloc[2]
        st.markdown("##### Default rate by score decile (1 = riskiest)")
        st.plotly_chart(ch.rate_bars(deciles, average=test.target_default.mean(), hover=["accounts"], height=380),
                        use_container_width=True, config=ch.CONFIG)
        st.caption(f"The riskiest 30% of accounts hold {top3:.0%} of all defaults.")
    with right:
        st.markdown("##### What drives the score (mean absolute contribution)")
        imp = reports["feature_importance"].head(12).rename(columns={"label": "segment"})
        st.plotly_chart(ch.rate_bars(imp, value=f"mean_abs_shap_{champion}", value_format=".3f", height=380),
                        use_container_width=True, config=ch.CONFIG)
        st.caption("SHAP values: for the logistic regression these are exact (coefficient x deviation from average).")

    st.markdown("#### Fairness")
    fair = reports["fairness"]
    fair["four-fifths test"] = fair.adverse_impact_ratio.map(lambda r: "Pass" if r >= 0.8 else "Review")
    st.dataframe(fair, hide_index=True, use_container_width=True, column_config={
        "group": "Group", "attribute": "Attribute", "accounts": "Accounts",
        "actual_default_rate": st.column_config.NumberColumn("Actual default rate", format="percent"),
        "avg_predicted_pd": st.column_config.NumberColumn("Avg predicted PD", format="percent"),
        "share_high_risk": st.column_config.NumberColumn("Rated High / Very High", format="percent"),
        "adverse_impact_ratio": st.column_config.NumberColumn("Adverse impact ratio", format="%.3f"),
        "four-fifths test": "Four-fifths test"})
    st.caption("Gender, marital status and age are not model inputs. Adverse impact ratio = share of the group not "
               "rated High/Very High, relative to the most favoured group (four-fifths rule: >= 0.80). Younger "
               "borrowers still fall below 0.80 through income and job tenure; this goes to Compliance for review.")
    checks = metrics.get("design_checks")
    if checks:
        age = checks["age"]
        st.caption(f"Age re-tested on the hold-out: with age as an input, test AUC is {age['test_auc_with']:.3f} "
                   f"(without: {age['test_auc_without']:.3f}) and the 21-25 ratio falls to "
                   f"{age['adverse_impact_ratio_test_with']['21-25']:.2f} (without: "
                   f"{age['adverse_impact_ratio_test_without']['21-25']:.2f}), so age stays out.")

    st.markdown("#### Monitoring after deployment")
    left, right = st.columns([1, 1.2])
    with left:
        champ = metrics["models"][champion]
        st.dataframe(pd.DataFrame([
            {"Check": "Score PSI, train vs test", "Value": round(champ["psi_train_vs_test"], 4)},
            {"Check": "Score PSI, 2018-21 vs 2022-23", "Value": round(champ["psi_development_vs_recent"], 4)}]),
            hide_index=True, use_container_width=True)
        st.dataframe(reports["csi_baseline"].drop(columns="feature").rename(
            columns={"label": "Top-10 feature", "csi_development_vs_recent": "CSI 2018-21 vs 2022-23",
                     "status": "Status"}), hide_index=True, use_container_width=True)
    with right:
        st.markdown(
            "- **Monthly**: PSI of the score distribution and CSI of the top 10 features against the development "
            "baseline. Below 0.10 is stable, 0.10-0.25 investigate, above 0.25 escalate to model risk.\n"
            "- **Quarterly**, once outcomes mature: AUC, KS and calibration by risk band against the validated "
            "baseline. Escalate if Gini falls more than 10% relative, or actual and predicted default rates "
            "diverge by more than 20% in any band.\n"
            "- **Retrain** annually, or earlier on a breach. LightGBM stays as a challenger and is re-tested at "
            "every retrain.\n"
            "- Fairness ratios and anomaly-review outcomes (confirm/dismiss rates) are re-checked quarterly. "
            "Every change goes through independent validation (SR 11-7).")

    st.markdown("#### Deliberately excluded from the model")
    st.dataframe(pd.DataFrame(metrics["excluded_features"].items(), columns=["Feature", "Reason"]), hide_index=True,
                 use_container_width=True)
    lf = metrics["logistic_features"]
    st.caption("Scorecard sign check (logistic regression): every feature must move risk in the expected direction. "
               f"Dropped because their coefficient pointed the wrong way: {', '.join(lf['dropped_by_sign_check'])}. "
               f"Left out as near-duplicates of a kept feature: {', '.join(lf['left_out_as_duplicates'])}. "
               "LightGBM uses all features with monotone constraints instead.")
    if checks:
        coll = checks["collections"]
        st.caption(f"Collections re-tested on the hold-out: adding its event counts and recoveries gives test AUC "
                   f"{coll['test_auc_with']:.3f} against {coll['test_auc_without']:.3f} without, so its unreliable "
                   "data stays out.")


PAGES = {
    "overview": st.Page(page_portfolio, title="Portfolio view", icon=":material/dashboard:", default=True),
    "segments": st.Page(page_segments, title="Segment drill-down", icon=":material/grid_view:"),
    "account": st.Page(page_account, title="Account-level view", icon=":material/person_search:"),
    "chat": st.Page(page_chat, title="Ask the Portfolio", icon=":material/forum:"),
    "model": st.Page(page_model, title="Model performance & monitoring", icon=":material/monitoring:"),
    "anomalies": st.Page(page_anomalies, title="Anomaly review", icon=":material/policy_alert:"),
}
# sidebar sections mirror the assignment brief
st.navigation({
    "Dashboard": [PAGES["overview"], PAGES["segments"], PAGES["account"]],
    "AI feature": [PAGES["chat"]],
    "Model": [PAGES["model"]],
    "Bonus AI": [PAGES["anomalies"]],
}).run()
