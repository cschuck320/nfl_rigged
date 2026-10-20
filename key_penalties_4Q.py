import nflreadpy as nfl
import numpy as np
import pandas as pd
import plotly.express as px

# 1. Load PBP data for 2024-2026
df = nfl.load_pbp([2024, 2025, 2026]).to_pandas()

# Filter for regular season standard quarters Q1-Q4
df_reg = df[(df["qtr"].isin([1, 2, 3, 4]))].copy()

# Define key judgment penalties (DPI, Roughing, Holding)
penalty_pattern = "Roughing the Passer|Offensive Pass Interference|Defensive Pass Interference|Offensive Holding|Defensive Holding"
df_reg["is_key_penalty"] = (df_reg["penalty"] == 1) & (
    df_reg["penalty_type"].str.contains(penalty_pattern, case=False, na=False)
)

# Categorize spreads for Q4 game-context analysis
df_reg["abs_spread"] = df_reg["spread_line"].abs()
df_reg["spread_category"] = np.select(
    [
        df_reg["abs_spread"] <= 3,
        (df_reg["abs_spread"] > 3) & (df_reg["abs_spread"] < 10),
        df_reg["abs_spread"] >= 10,
    ],
    [
        "Close Spread (<= 3 pts)",
        "Moderate Spread (3.5 - 9.5 pts)",
        "Blowout Spread (>= 10 pts)",
    ],
    default="Other",
)

# ==============================================================================
# ANALYSIS 1: Overall Penalty Frequency by Quarter (Q1 vs Q2 vs Q3 vs Q4)
# ==============================================================================

qtr_summary = (
    df_reg.groupby("qtr")
    .agg(
        total_plays=("play_id", "count"),
        total_flags=("penalty", "sum"),
        key_flags=("is_key_penalty", "sum"),
    )
    .reset_index()
)

# Calculate rates per 100 plays
qtr_summary["all_flags_per_100_plays"] = (
    qtr_summary["total_flags"] / qtr_summary["total_plays"]
) * 100
qtr_summary["key_flags_per_100_plays"] = (
    qtr_summary["key_flags"] / qtr_summary["total_plays"]
) * 100

print("--- Overall Quarterly Penalty Rates (2024-2026) ---")
print(qtr_summary.to_string(index=False))

# ==============================================================================
# ANALYSIS 2: Q4 Penalty Rate Breakdown by Pre-Game Spread (Close vs Blowout)
# ==============================================================================

q4_context = (
    df_reg[df_reg["spread_category"].isin(["Close Spread (<= 3 pts)", "Blowout Spread (>= 10 pts)"])]
    .groupby(["qtr", "spread_category"])
    .agg(
        total_plays=("play_id", "count"),
        total_flags=("penalty", "sum"),
        key_flags=("is_key_penalty", "sum"),
    )
    .reset_index()
)

q4_context["key_flags_per_100_plays"] = (
    q4_context["key_flags"] / q4_context["total_plays"]
) * 100

print("\n--- Key Penalty Rates by Quarter & Spread Category ---")
print(q4_context.pivot(index="qtr", columns="spread_category", values="key_flags_per_100_plays"))

# ==============================================================================
# VISUALIZATION: Plotly Bar Chart
# ==============================================================================

fig = px.bar(
    qtr_summary,
    x="qtr",
    y=["all_flags_per_100_plays", "key_flags_per_100_plays"],
    barmode="group",
    title="NFL Penalty Rate per 100 Plays by Quarter (2024-2026)",
    labels={
        "qtr": "Quarter",
        "value": "Flags per 100 Plays",
        "variable": "Penalty Category",
    },
    template="plotly_white",
)

# Rename legend labels cleanly
new_names = {
    "all_flags_per_100_plays": "All Penalty Flags",
    "key_flags_per_100_plays": "Key Flags (DPI, Roughing, Holding)",
}
fig.for_each_trace(lambda t: t.update(name=new_names[t.name]))

fig.update_xaxes(type="category", tickvals=[1, 2, 3, 4], ticktext=["Q1", "Q2", "Q3", "Q4"])
fig.write_html("quarterly_penalty_uptick.html")

print("\nSaved quarterly plot to 'quarterly_penalty_uptick.html'")