import nflreadpy as nfl
import numpy as np
import pandas as pd
import plotly.express as px

pd.set_option("display.max_columns", None)

# Load PBP Data & Game Schedules (2016–2026)
years = list(range(2016, 2027))
df = nfl.load_pbp(years).to_pandas()

# Load schedule metadata to map Head Referees (Crews) by game_id
schedules = nfl.load_schedules(years).to_pandas()
ref_map = schedules[["game_id", "referee"]].drop_duplicates().dropna(subset=["referee"])
print(ref_map.head(10))
# Filter regular season plays with valid score differentials and spreads
df_reg = df[
    (df["season_type"] == "REG")
    & (df["qtr"].isin([1, 2, 3, 4]))
    & (df["score_differential"].notna())
    & (df["spread_line"].notna())
    & (df["posteam"].notna())
].copy()

# Merge referee data onto play-by-play
df_reg = df_reg.merge(ref_map, on="game_id", how="left")
df_reg = df_reg[df_reg["referee"].notna()].copy()

# 2. Key Judgment Penalties (DPI, Roughing, Holding)
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

# 4. Identify Penalty Beneficiary Team & Cover Margin
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

# Filter for key penalty calls on non-push plays
key_flags_df = df_reg[
    (df_reg["is_key_penalty"] == 1)
    & (df_reg["beneficiary_cover_margin"] != 0)
].copy()

# ==============================================================================
# CREW ANALYSIS A: Overall Key Flag Bias by Referee Crew
# ==============================================================================

games_per_ref = df_reg.groupby("referee")["game_id"].nunique().rename("games_officiated")

ref_overall = (
    key_flags_df.groupby("referee")
    .agg(
        total_key_flags=("play_id", "count"),
        flags_benefiting_trailing=(
            "beneficiary_cover_status",
            lambda x: (x == "Benefited Team Trailing Spread").sum(),
        ),
    )
)

ref_overall["pct_trailing_all"] = (
    ref_overall["flags_benefiting_trailing"] / ref_overall["total_key_flags"]
) * 100

# ==============================================================================
# CREW ANALYSIS B: Q4 Close-Spread Key Flag Bias by Referee Crew
# ==============================================================================

q4_close_flags = key_flags_df[
    (key_flags_df["qtr"] == 4)
    & (key_flags_df["posteam_cover_margin"].abs() <= 8)
]

ref_q4_close = (
    q4_close_flags.groupby("referee")
    .agg(
        q4_close_flags=("play_id", "count"),
        q4_flags_benefiting_trailing=(
            "beneficiary_cover_status",
            lambda x: (x == "Benefited Team Trailing Spread").sum(),
        ),
    )
)

ref_q4_close["pct_trailing_q4_close"] = (
    ref_q4_close["q4_flags_benefiting_trailing"] / ref_q4_close["q4_close_flags"]
) * 100

# ==============================================================================
# CREW ANALYSIS C: Merge & Filter by Minimum Sample Size (e.g. >= 30 Games)
# ==============================================================================

crew_summary = (
    pd.concat([games_per_ref, ref_overall, ref_q4_close], axis=1)
    .fillna(0)
    .reset_index()
)

crew_summary["flags_per_game"] = (
    crew_summary["total_key_flags"] / crew_summary["games_officiated"]
).round(2)

MIN_GAMES = 30
filtered_crews = crew_summary[crew_summary["games_officiated"] >= MIN_GAMES].sort_values(
    by="pct_trailing_all", ascending=False
)

print(f"--- Officiating Crew Bias Summary (Min. {MIN_GAMES} Games, 2016–2026) ---")
print(
    filtered_crews[
        [
            "referee",
            "games_officiated",
            "total_key_flags",
            "flags_per_game",
            "pct_trailing_all",
            "pct_trailing_q4_close",
        ]
    ].round(2).to_string(index=False)
)

# ==============================================================================
# VISUALIZATION 1: Horizontal Bar Chart — Overall Crew Spread Bias
# ==============================================================================

fig_bar = px.bar(
    filtered_crews.sort_values(by="pct_trailing_all"),
    x="pct_trailing_all",
    y="referee",
    orientation="h",
    text_auto=".1f",
    title=f"Officiating Crew Bias: % Key Flags Benefiting Trailing Team (2016–2026, Min. {MIN_GAMES} Games)",
    labels={
        "pct_trailing_all": "% Key Flags Benefiting Trailing Team",
        "referee": "Head Official (Crew)",
    },
    template="plotly_white",
    color="pct_trailing_all",
    color_continuous_scale="RdYlBu",
)

fig_bar.add_vline(
    x=50.0,
    line_dash="dash",
    line_color="black",
    annotation_text="50% Neutral Baseline",
    annotation_position="top right",
)

fig_bar.write_html("referee_crew_spread_bias.html")
print("\nSaved crew bar chart to 'referee_crew_spread_bias.html'")

# ==============================================================================
# VISUALIZATION 2: Scatter Plot — Overall vs. High-Leverage Q4 Close-Spread Bias
# ==============================================================================

fig_scatter = px.scatter(
    filtered_crews,
    x="pct_trailing_all",
    y="pct_trailing_q4_close",
    size="total_key_flags",
    hover_name="referee",
    text="referee",
    title="Referee Crew Bias: Overall vs. Q4 Close-Spread Trailing Benefit (%)",
    labels={
        "pct_trailing_all": "Overall % Flags Benefiting Trailing Team",
        "pct_trailing_q4_close": "Q4 Close-Spread % Flags Benefiting Trailing Team",
        "total_key_flags": "Total Key Flags",
    },
    template="plotly_white",
)

fig_scatter.add_hline(y=50.0, line_dash="dash", line_color="gray")
fig_scatter.add_vline(x=50.0, line_dash="dash", line_color="gray")
fig_scatter.update_traces(textposition="top center")

fig_scatter.write_html("referee_crew_scatter.html")
print("Saved crew scatter plot to 'referee_crew_scatter.html'")