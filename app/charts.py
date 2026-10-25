"""
Plotly chart builders for the dashboard. One visual system throughout:
  - a single series is always blue; two compared series are blue and orange (validated pair)
  - risk bands and payment states use the fixed status colours, always next to a text label
  - heatmaps diverge blue (below average) / grey (average) / red (above average)
  - thin bars with rounded ends, hairline solid grid, one y-axis per chart, hover on every mark
"""
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAISES, LOWERS = "#e34948", "#2a78d6"
STATUS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#ec835a", "Very High": "#d03b3b",
          "On-Time": "#0ca30c", "Late": "#fab219", "Missed": "#d03b3b"}
DIVERGING = [[0.0, "#1c5cab"], [0.25, "#86b6ef"], [0.5, "#f0efec"], [0.75, "#f1a09f"], [1.0, "#b8302f"]]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
CONFIG = {"displayModeBar": False}


def money(value):
    for unit, size in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(value) >= size:
            return f"${value / size:,.1f}{unit}"
    return f"${value:,.0f}"


def _style(fig, height=340, legend=False, horizontal=False, pct_axis=True, top=None):
    fig.update_layout(
        height=height, margin=dict(l=4, r=24, t=top or (36 if legend else 12), b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=INK_2),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, size=13, color=INK)),
        barcornerradius=4, bargap=0.4, bargroupgap=0.1,
    )
    value_axis = dict(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, showline=False,
                      tickformat=".0%" if pct_axis else None, rangemode="tozero")
    category_axis = dict(showgrid=False, showline=True, linecolor=AXIS, ticks="", zeroline=False)
    if horizontal:
        fig.update_xaxes(**value_axis)
        fig.update_yaxes(**category_axis, autorange="reversed")
    else:
        fig.update_yaxes(**value_axis)
        fig.update_xaxes(**category_axis)
    return fig


def rate_bars(table, label="segment", value="default_rate", average=None, horizontal=True, hover=None,
              value_format=".1%", height=None):
    """One series: a rate per segment, value at the bar tip, optional average reference line."""
    text = [format(v, value_format) for v in table[value]]
    custom = table[hover].to_numpy() if hover else None
    hovertemplate = "<b>%{" + ("y" if horizontal else "x") + "}</b><br>" + \
        ("%{x:" if horizontal else "%{y:") + value_format + "}"
    if hover:
        hovertemplate += "".join(f"<br>{h.replace('_', ' ')}: %{{customdata[{i}]:,}}" for i, h in enumerate(hover))
    bar = go.Bar(x=table[value] if horizontal else table[label], y=table[label] if horizontal else table[value],
                 orientation="h" if horizontal else "v", marker_color=BLUE, text=text, textposition="outside",
                 cliponaxis=False, textfont=dict(color=INK_2), customdata=custom,
                 hovertemplate=hovertemplate + "<extra></extra>")
    fig = go.Figure(bar)
    if average is not None:
        line = dict(line_width=1, line_color=INK_2, annotation_text=f"average {average:{value_format}}",
                    annotation_font=dict(color=INK_2, size=12))
        if horizontal:
            fig.add_vline(x=average, annotation_position="top", **line)
        else:
            fig.add_hline(y=average, annotation_position="top left", **line)
    rows = len(table)
    # room above the plot for the "average" label of a vertical reference line
    fig = _style(fig, height=height or (max(220, 44 * rows + 40) if horizontal else 340), horizontal=horizontal,
                 pct_axis=value_format.endswith("%"), top=32 if average is not None and horizontal else None)
    return fig


def share_bars(table, label="segment"):
    """Exposure share next to share of defaulted exposure: where the money is vs where the losses are."""
    fig = go.Figure()
    for column, name, color in [("exposure_share", "Share of exposure", BLUE),
                                ("share_of_defaulted_exposure", "Share of defaulted exposure", ORANGE)]:
        fig.add_bar(y=table[label], x=table[column], name=name, orientation="h", marker_color=color,
                    hovertemplate="<b>%{y}</b><br>" + name + ": %{x:.1%}<extra></extra>")
    rows = len(table)
    return _style(fig, height=max(240, 56 * rows + 60), legend=True, horizontal=True)


def band_bars(table, value, label="segment", value_format=".1%", pct_axis=True):
    """Risk bands in their status colours; the band name is always on the axis next to the bar."""
    colors = [STATUS.get(s, BLUE) for s in table[label]]
    text = [format(v, value_format) if value_format != "money" else money(v) for v in table[value]]
    fig = go.Figure(go.Bar(x=table[label], y=table[value], marker_color=colors, text=text, textposition="outside",
                           cliponaxis=False, textfont=dict(color=INK_2),
                           hovertemplate="<b>%{x}</b><br>%{text}<extra></extra>"))
    fig = _style(fig, height=300, pct_axis=pct_axis)
    fig.update_layout(bargap=0.65)
    return fig


def trend_lines(table, x, series, height=320):
    """Rates over time; `series` maps a column to its legend name (at most two, blue then orange)."""
    fig = go.Figure()
    for (column, name), color in zip(series.items(), [BLUE, ORANGE]):
        fig.add_scatter(x=table[x].astype(str), y=table[column], name=name, mode="lines+markers",
                        line=dict(color=color, width=2), marker=dict(size=8, color=color, line=dict(width=2,
                                                                                                color="white")),
                        hovertemplate="%{x}<br>" + name + ": %{y:.1%}<extra></extra>")
    fig = _style(fig, height=height, legend=len(series) > 1)
    fig.update_xaxes(type="category")
    return fig


def heatmap(rate, count, average, row_label, col_label):
    """Default rate per cell, coloured by distance from the average; small cells are marked with *."""
    z = rate.to_numpy(dtype=float)
    spread = np.nanmax(np.abs(z - average)) or 0.01
    text, ink = [], []
    for i in range(z.shape[0]):
        row_text = []
        for j in range(z.shape[1]):
            n = count.iloc[i, j]
            row_text.append("" if np.isnan(z[i, j]) else f"{z[i, j]:.1%}{'*' if n < 100 else ''}<br>n={n:,}")
        text.append(row_text)
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(c) for c in rate.columns], y=[str(r) for r in rate.index], text=text,
        texttemplate="%{text}", textfont=dict(size=12), colorscale=DIVERGING, zmid=average,
        zmin=average - spread, zmax=average + spread, xgap=2, ygap=2,
        customdata=count.to_numpy(),
        hovertemplate=f"{row_label}: %{{y}}<br>{col_label}: %{{x}}<br>default rate %{{z:.1%}}"
                      "<br>accounts %{customdata:,}<extra></extra>",
        colorbar=dict(title=dict(text="default rate", side="right"), tickformat=".0%", thickness=12, outlinewidth=0,
                      tickvals=[t for t in np.arange(0, 1.01, 0.1) if average - spread <= t <= average + spread])))
    fig.update_layout(height=max(320, 64 * len(rate.index) + 80), margin=dict(l=4, r=4, t=12, b=4),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT, size=13, color=INK_2))
    fig.update_xaxes(side="top", showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def driver_bars(drivers):
    """Contribution of each factor to this account's risk vs an average account (log-odds)."""
    labels = [f"{d['label']} ({d['value']})" for d in drivers]
    fig = go.Figure()
    for name, color, keep in [("Raises risk", RAISES, lambda v: v > 0), ("Lowers risk", LOWERS, lambda v: v <= 0)]:
        fig.add_bar(y=labels, x=[d["impact"] if keep(d["impact"]) else None for d in drivers], name=name,
                    orientation="h", marker_color=color,
                    hovertemplate="%{y}<br>" + name.lower() + ": %{x:+.2f} log-odds<extra></extra>")
    fig = _style(fig, height=max(260, 40 * len(drivers) + 60), legend=True, horizontal=True, pct_axis=False)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(zeroline=True, zerolinecolor=AXIS, zerolinewidth=1, rangemode="normal",
                     title=dict(text="effect on risk (log-odds)", font=dict(size=12, color=MUTED)))
    return fig


def repayment_bars(history):
    """Days past due for each of the 12 months, coloured by payment status."""
    fig = go.Figure()
    for status in ["On-Time", "Late", "Missed"]:
        part = history[history.payment_status.eq(status)]
        fig.add_bar(x=part.payment_month, y=part.days_past_due.clip(lower=0.6), name=status,
                    marker_color=STATUS[status], customdata=part[["days_past_due", "emi_paid", "emi_due"]].to_numpy(),
                    hovertemplate="month %{x}<br>" + status + "<br>%{customdata[0]} days past due"
                                  "<br>paid $%{customdata[1]:,.0f} of $%{customdata[2]:,.0f}<extra></extra>")
    fig = _style(fig, height=280, legend=True, pct_axis=False)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(tickmode="linear", dtick=1, title=dict(text="month since origination",
                                                            font=dict(size=12, color=MUTED)))
    fig.update_yaxes(title=dict(text="days past due", font=dict(size=12, color=MUTED)))
    return fig


def roc_chart(y, predictions):
    """ROC curves; `predictions` maps a model name to its scores (two models: blue, orange)."""
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=AXIS, width=1), name="random",
                    hoverinfo="skip", showlegend=False)
    for (name, p), color in zip(predictions.items(), [BLUE, ORANGE]):
        fpr, tpr, _ = roc_curve(y, p)
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=name, line=dict(color=color, width=2),
                        hovertemplate="false positive rate %{x:.0%}<br>true positive rate %{y:.0%}<extra></extra>")
    fig = _style(fig, height=340, legend=True)
    fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=GRID, title=dict(text="false positive rate",
                                                                                font=dict(size=12, color=MUTED)))
    fig.update_yaxes(title=dict(text="true positive rate", font=dict(size=12, color=MUTED)))
    return fig


def paired_bars(table, label, first, second, names, value_format=".1%"):
    """Two measures in the same unit side by side (e.g. predicted vs actual default rate)."""
    fig = go.Figure()
    for column, name, color in [(first, names[0], BLUE), (second, names[1], ORANGE)]:
        fig.add_bar(x=table[label].astype(str), y=table[column], name=name, marker_color=color,
                    hovertemplate="<b>%{x}</b><br>" + name + ": %{y:" + value_format + "}<extra></extra>")
    return _style(fig, height=320, legend=True)
