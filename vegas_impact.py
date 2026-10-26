import nflreadpy as nfl
import numpy as np
import pandas as pd
import plotly.express as px

pd.set_option("display.max_columns", None)

START_DATE = 2016
# Load PBP Data and convert to Pandas
years = list(range(START_DATE, 2027))
df = nfl.load_pbp(years).to_pandas()

# Filter regular season plays with valid score differentials and spreads
df_reg = df[
    (df["game_type"] == "REG")
    & df["qtr"].isin([1, 2, 3, 4])
    & (df["score_differential"].notna())
    & (df["spread_line"].notna())
    & (df["posteam"].notna())
].copy()

# Key Judgment Penalties (PI, Roughing, Holding)
penalty_pattern = "Roughing the Passer|Pass Interference|Holding"
df_reg["is_key_penalty"] = (df_reg["penalty"] == 1) & (
    df_reg["penalty_type"].str.contains(penalty_pattern, case=False, na=False)
)

# 3. Calculate Live Cover Margin for Possession Team (posteam)
df_reg["posteam_cover_margin"] = np.where(
    df_reg["posteam"] == df_reg["home_team"],
    df_reg["score_differential"] - df_reg["spread_line"],
    df_reg["score_differential"] + df_reg["spread_line"],
)

df_reg["posteam_cover_status"] = np.where(
    df_reg["posteam_cover_margin"] < 0,
    "Trailing Spread (Failing to Cover)",
    np.where(
        df_reg["posteam_cover_margin"] > 0,
        "Leading Spread (Covering)",
        "On Spread Line (Push)",
    ),
)

# 4. Identify Penalty Beneficiary Team
df_reg["beneficiary_team"] = np.where(
    df_reg["penalty_team"] == df_reg["defteam"],
    df_reg["posteam"],
    df_reg["defteam"],
)

df_reg["beneficiary_cover_margin"] = np.where(
    df_reg["beneficiary_team"] == df_reg["home_team"],
    (df_reg["home_score"] - df_reg["away_score"]) - df_reg["spread_line"],
    (df_reg["away_score"] - df_reg["home_score"]) + df_reg["spread_line"],
)

df_reg["beneficiary_cover_status"] = np.where(
    df_reg["beneficiary_cover_margin"] < 0,
    "Benefited Team Trailing Spread",
    "Benefited Team Leading Spread",
)

# ==============================================================================
# ANALYSIS A: Yearly Trend in Q4 Pass-Normalized Key Flag Rates
# ==============================================================================
q4_pass_plays = df_reg[(df_reg["play_type"] == "pass") & (df_reg["qtr"] == 4)]

q4_yearly_pass_rates = (
    q4_pass_plays.groupby(["season", "posteam_cover_status"])
    .agg(
        pass_plays=("play_id", "count"),
        key_flags=("is_key_penalty", "sum"),
    )
    .reset_index()
)

q4_yearly_pass_rates["key_flags_per_100_passes"] = (
    q4_yearly_pass_rates["key_flags"] / q4_yearly_pass_rates["pass_plays"]
) * 100

q4_pivot = q4_yearly_pass_rates.pivot(
    index="season",
    columns="posteam_cover_status",
    values="key_flags_per_100_passes",
).round(2)

print("--- Q4 Pass-Normalized Key Flag Rates by Season ---")
print(q4_pivot)

# ==============================================================================
# ANALYSIS B & C: Yearly Trend in Beneficiary Skew (% Benefiting Trailing Team)
# ==============================================================================
key_flags_df = df_reg[
    (df_reg["is_key_penalty"] == 1)
    & (df_reg["beneficiary_cover_margin"] != 0)
].copy()

# 1. Overall % Benefited Trailing by Season
overall_yearly_ben = (
    key_flags_df.groupby(["season", "beneficiary_cover_status"])
    .size()
    .unstack(fill_value=0)
)
overall_yearly_ben["pct_benefited_trailing_all_qtrs"] = (
    overall_yearly_ben["Benefited Team Trailing Spread"]
    / overall_yearly_ben.sum(axis=1)
) * 100

# 2. Q4 Close-Spread (Within 8 pts) % Benefited Trailing by Season
q4_close_flags = key_flags_df[
    (key_flags_df["qtr"] == 4)
    & (key_flags_df["posteam_cover_margin"].abs() <= 8)
]

q4_yearly_ben = (
    q4_close_flags.groupby(["season", "beneficiary_cover_status"])
    .size()
    .unstack(fill_value=0)
)
q4_yearly_ben["pct_benefited_trailing_q4_close"] = (
    q4_yearly_ben["Benefited Team Trailing Spread"]
    / q4_yearly_ben.sum(axis=1)
) * 100

# Merge yearly summaries into a single trend DataFrame
yearly_trends = pd.DataFrame(
    {
        "Overall % Flags Benefiting Trailing Team": overall_yearly_ben[
            "pct_benefited_trailing_all_qtrs"
        ],
        "Q4 Close-Spread % Flags Benefiting Trailing Team": q4_yearly_ben[
            "pct_benefited_trailing_q4_close"
        ],
    }
).reset_index()

print(
    "\n--- Yearly Beneficiary Trends (% of Key Flags Benefiting Trailing Team) ---"
)
print(yearly_trends.round(2).to_string(index=False))

# ==============================================================================
# VISUALIZATION: Multi-Line Plotly Chart against 50% Neutral Baseline
# ==============================================================================
fig = px.line(
    yearly_trends,
    x="season",
    y=[
        "Overall % Flags Benefiting Trailing Team",
        "Q4 Close-Spread % Flags Benefiting Trailing Team",
    ],
    markers=True,
    title="Yearly Trend: % of Key Flags Benefiting Team Trailing Spread (2016–2026)",
    labels={
        "season": "Season",
        "value": "Percentage of Key Flags (%)",
        "variable": "Metric",
    },
    template="plotly_white",
)

# Reference line representing expected 50% neutral baseline
fig.add_hline(
    y=50.0,
    line_dash="dash",
    line_color="red",
    annotation_text="50% Expected Baseline (No Bias)",
    annotation_position="bottom right",
)

fig.update_xaxes(dtick=1)
fig.write_html(f"yearly_spread_bias_trend_{START_DATE}_present.html")
print(f"\nSaved trend line chart to 'yearly_spread_bias_trend_{START_DATE}_present.html'")