# 5. AI feature: "Ask the Portfolio" chat

The brief asks for one AI feature, embedded in the dashboard. This build's AI feature is **"Ask the Portfolio"**:
natural-language Q&A over the dashboard data, answered by an LLM and grounded in the actual underlying data, so no
number is invented. It sits in the dashboard under **AI feature → Ask the Portfolio** (`src/chat.py`, page in
`app/app.py`). A second AI feature, anomaly detection, is included as a bonus at the end of this document.

## What it does

An analyst types a question such as the brief's own example, *"Which segment has the highest default rate this
quarter?"* (the first suggested question on the page). **NVIDIA Nemotron-3 Ultra** (fallback: Nemotron-3.5 Lightning)
answers it through NVIDIA's OpenAI-compatible API. For that example question the assistant filters to the latest
origination quarter in the data (2023 Q4), ranks business segments (loan type, income band, employment, region, age
band and more) by default rate, and warns when a segment is too small to trust.

## How it stays grounded (no hallucinated numbers)

1. The model **never sees raw data and never calculates.** It can only call 8 query tools: overview, segment breakdown,
   highest-default segments, cross-tab, vintage trend, account lookup, top accounts and model info. Each tool computes
   its numbers with `src/portfolio.py`, the same code the dashboard uses, and returns ready-made comparisons (lift,
   gap to the portfolio, share of exposure) so the model has no arithmetic to do.
2. Allowed filter values come from the data itself, so the model can't filter on a segment that doesn't exist.
3. **Every number in the answer is checked twice** (`verify` in `src/chat.py`). First, it must exist in the tool
   results, allowing for rounding and units ("$314.5 million", "52.3%", "1.7x"). Second, if it belongs to a specific
   segment, cell or account, its sentence must be about that segment, so a real figure quoted for the wrong segment
   is caught too ("South defaults at 24.7%", when 24.7% is the mortgage rate). Numbers that fail are shown to the
   user as a warning; a number that only repeats the question ("Is the default rate above 20%?") is labelled as
   such, and each verified number lists where it came from.
4. Each answer shows the exact queries and tables behind it.
5. Without an API key, or if the service fails, a rule-based router reads the question (filters such as
   "unemployed", one or two dimensions, rankings, trends, an account id), calls the same tools and runs the same
   check, so every suggested question still gets a grounded answer.

**Tested live** on 16 questions (the suggested ones plus forecasts, rankings, comparisons and a yes/no question), run
repeatedly: numbers copied from the tools pass, and every flag was a number the model had produced itself instead of
copying it, such as a difference between two rates ("13.4 percentage points higher", flagged in every run), a rounded
range ("roughly 2,400–2,500 accounts") or an invented threshold ("fewer than 60 accounts"). In the final run 13 of the
16 answers verified in full. Two kinds of answer the check cannot judge, and the analyst still must: a real number
described too broadly ("every region defaults at 16.1%", which is the portfolio rate), and a number the model
calculated correctly, which is flagged anyway because the tools didn't return it. `tests/test_chat.py` keeps these
behaviours fixed.

## What triggers it

A user's question, typed in the chat box or picked from the suggested questions. Optionally the question is limited to
the dashboard's current sidebar filters, so it answers over exactly the data on screen. Nothing runs in the background.

## What a false positive looks like

A confident answer the data can't really support: for example reading "this quarter" in a way the user didn't mean,
applying the wrong filter, or quoting a segment rate built on very few accounts. Mitigations: the period used is always
stated, small segments are flagged, the queries behind every answer are shown, and numbers that are not in the data,
or sit next to the wrong segment, are flagged.

## What a false negative looks like

"I can't answer that" when the data could answer, usually because of phrasing, or a question answered at a coarser cut
than asked. Mitigations: suggested questions show what works, and the same cut is one click away in the Portfolio view
and Segment drill-down.

## The human-in-the-loop step

Answers are for exploration, not decisions. Before a figure goes into a decision or a report, the analyst checks it in
the "data behind this answer" panel. Every answer can be rated (thumbs up or down), which is logged to
`data/review/chat_feedback.csv`; low-rated answers are reviewed monthly to improve the tools and instructions.

## Data protection

In this prototype, tool results (aggregates and, for account questions, one account's details) go to NVIDIA's hosted
API. That is acceptable only because the data is synthetic. In production, Nemotron runs as an **NVIDIA NIM inside the
bank's network**, so no customer data reaches a third party.

---

# Bonus AI feature: anomaly detection

Beyond the brief's one AI feature. Dashboard: **Bonus AI → Anomaly review** (`src/anomaly.py`). It flags accounts
whose repayment, bureau or affordability pattern is unusual **for their peer group** (same loan type and income band) or
**against their own history**, and explains each flag in plain words.

## What triggers it

Runs with every monthly data refresh. An account is flagged if any trigger applies:

| Trigger | Rule | Severity | Accounts | Default rate |
|---|---|---|---|---|
| Peer outlier | Isolation Forest on each account's percentile within its peer group for 10 risk metrics (repayment, bureau, affordability), counting only the risky direction; the 5% most unusual are flagged | High for the top 1%, otherwise Medium | 735 | 33.2% |
| Affordability breach | Monthly instalment larger than the borrower's monthly income | High | 79 | 45.6% |
| Payment shock (own history) | 2+ missed payments in the last 3 months after 9 months with no missed payment | Medium | 68 | 22.1% |
| **All flags** | | | **815 (5.5%)** | **32.4% (2.0x the 16.1% average)** |

High-severity flags (205) default at 47.3%. Each flag carries reasons such as *"Average days past due (12 months) 15.6
vs peer median 4.2 (higher than 92% of peers)"*. The target is never used to build flags; it is used only to check that
flags concentrate real risk. 80% of flags are accounts the PD model already rates High or Very High: the value is the
*explanation* against peers and the *policy breaches*, not a second default model.

## False positives and false negatives

- **False positive:** a flag on an account that keeps paying, e.g. a high loan-to-income for a customer whose rental
  income isn't recorded, or late payments caused by a payment-system outage. About two thirds of flags don't default,
  so a flag is a prompt to look, not a verdict.
- **False negative:** a default that looked normal, e.g. a borrower with a clean first year who loses their job later.
  The detector catches about 1 in 9 defaults (recall 11%); the PD model covers the whole book.

## Human in the loop

Analysts work the queue (High severity first) and **confirm** (escalate to an early-intervention contact) or
**dismiss** with a reason (data error, known customer circumstance, benign pattern). Each decision is logged with name
and time in `data/review/anomaly_reviews.csv`; dismissal reasons are reviewed monthly to tune thresholds.
