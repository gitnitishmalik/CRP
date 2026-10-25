"""Draws docs/architecture.png: ingestion -> analytics -> model -> dashboard (+ AI feature), tools at each stage.

    python docs/make_architecture.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "architecture.png"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
BLUE, BLUE_BG = "#2a78d6", "#f2f6fd"
ORANGE, ORANGE_BG = "#eb6834", "#fcf2ed"
GREY, GREY_BG = "#c3c2b7", "#f5f4f0"
W, H = 1900, 1124  # canvas in pixels; y grows downwards
LINE = 33          # line height of body text


def box(ax, x, y, w, h, title, lines, edge, face, bold_first=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=16", linewidth=2,
                                edgecolor=edge, facecolor=face))
    ax.text(x + 22, y + 42, title, fontsize=17, fontweight="bold", color=INK, va="center")
    for i, line in enumerate(lines):
        bold = bold_first and i == 0
        ax.text(x + 22, y + 92 + i * LINE, line, fontsize=13.2, color=INK if bold else INK_2, va="center",
                fontweight="bold" if bold else "normal")


def arrow(ax, start, end, color=INK_2, label=None, label_at=None, ha="left"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=2, color=color,
                                 shrinkA=0, shrinkB=0))
    if label:
        ax.text(*label_at, label, fontsize=12.5, color=MUTED, va="center", ha=ha)


fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(28, 34, "Architecture: ingestion → analytics → model → dashboard (+ AI feature)", fontsize=24,
        fontweight="bold", color=INK, va="center")
ax.text(28, 78, "Every number in the dashboard and the chat comes from the same code (src/portfolio.py) over the same "
                "cleaned data.", fontsize=14.5, color=INK_2, va="center")

top, top_h = 130, 300
box(ax, 24, top, 250, top_h, "Source data",
    ["5 CSV tables:", "customers", "loan_accounts", "bureau_credit", "repayment_history", "collections_events"],
    GREY, GREY_BG)
box(ax, 316, top, 360, top_h, "1  Ingestion & cleaning",
    ["src/ingest.py", "Python, pandas, SciPy", "45 checks, all logged", "aggregate 1:N tables, then join",
     "→ accounts_clean.csv, dq_log.csv"], BLUE, BLUE_BG, bold_first=True)
box(ax, 718, top, 340, top_h, "2  Analytics",
    ["src/portfolio.py", "pandas", "profiles, cross-cuts, trends", "→ notebook 02", "shared by dashboard + chat"],
    BLUE, BLUE_BG, bold_first=True)
box(ax, 1100, top, 370, top_h, "3  Default model",
    ["src/train_model.py", "scikit-learn, LightGBM, SHAP", "logistic champion vs LightGBM",
     "grouped CV + out-of-time test", "→ scores.csv, models/, reports/"], BLUE, BLUE_BG, bold_first=True)
box(ax, 1512, top, 364, top_h, "4  Dashboard",
    ["app/app.py", "Streamlit, Plotly", "portfolio view, segment drill-down,", "account view, model page",
     "6 pages, shared filters"], INK, "white", bold_first=True)
for x0, x1 in [(274, 316), (676, 718), (1058, 1100), (1470, 1512)]:
    arrow(ax, (x0 + 4, top + top_h / 2), (x1 - 4, top + top_h / 2))

low, low_h = 520, 290
box(ax, 24, low, 652, low_h, "Human in the loop",
    ["Analysts confirm or dismiss every anomaly flag", "(decision, reason, name and time are logged).",
     "Chat answers show the queries and the source of", "every number, and are rated; ratings are logged.",
     "No automated action on any customer."], GREY, GREY_BG)
box(ax, 718, low, 340, low_h, "Model monitoring",
    ["monthly: score PSI, feature CSI", "quarterly: AUC, KS, calibration,", "fairness ratios",
     "annual retrain + challenger", "SR 11-7 validation"], GREY, GREY_BG)
box(ax, 1100, low, 370, low_h, "Bonus AI: anomalies",
    ["src/anomaly.py", "Isolation Forest on peer", "percentiles + 2 rules", "→ flags with plain reasons",
     "→ analyst review queue"], ORANGE, ORANGE_BG, bold_first=True)
box(ax, 1512, low, 364, low_h, "5  AI: Ask the Portfolio",
    ["src/chat.py", "LLM limited to 8 query tools", "every number checked against", "tool results and segments",
     "offline rule-based fallback"], ORANGE, ORANGE_BG, bold_first=True)

llm = 880
box(ax, 1512, llm, 364, 224, "NVIDIA API",
    ["Nemotron-3 Ultra", "fallback: Nemotron-3.5 Lightning", "sees tool results only", "prod: NIM inside the bank"],
    GREY, GREY_BG, bold_first=True)

arrow(ax, (1285, top + top_h + 4), (1285, low - 4), ORANGE, "scores", (1300, 475))
arrow(ax, (1462, low - 4), (1528, top + top_h + 6), ORANGE, "flags", (1470, 470), ha="right")
arrow(ax, (1760, low - 4), (1760, top + top_h + 4), ORANGE, "answers", (1775, 475))
arrow(ax, (888, low - 4), (1108, top + top_h + 6), GREY, "monitors", (872, 488), ha="right")
arrow(ax, (1630, low + low_h + 4), (1630, llm - 4), ORANGE, "tool calls", (1615, 845), ha="right")
arrow(ax, (1770, llm - 4), (1770, low + low_h + 4), ORANGE, "results", (1785, 845))

fig.savefig(OUT, dpi=100, facecolor="white")
print("saved", OUT)
