import nflreadpy as nfl
import numpy as np
import pandas as pd
import plotly.express as px

# Load 2024-2025 Play-by-Play Data
years = list(range(2024, 2027))
df = nfl.load_pbp(years).to_pandas()

# # Filter out non-regular/postseason noise if necessary
# df = df[df["game_type"] == "REG"].copy()

# ==============================================================================
# ANALYSIS 1: Game-Level Total Flags vs. Vegas Total Line & Spread Magnitude
# ==============================================================================

# Aggregate total flags per game
game_stats = (
    df.groupby(["game_id", "spread_line", "total_line"])
    .agg(
        total_flags=("penalty", "sum"),
        abs_spread=("spread_line", lambda x: abs(x.iloc[0])),
    )
    .reset_index()
)

# Calculate Pearson correlation matrix
corr_matrix = game_stats[
    ["total_flags", "total_line", "abs_spread"]
].corr(method="pearson")
print("--- Game-Level Correlation Matrix ---")
print(corr_matrix)

# Plot Game-Level Correlation: Flags vs Over/Under Line
fig_game = px.scatter(
    game_stats,
    x="total_line",
    y="total_flags",
    trendline="ols",
    title="NFL Total Flags vs. Vegas Over/Under Line",
    labels={
        "total_line": "Vegas Total Line (Points)",
        "total_flags": "Total Flags Thrown",
    },
    hover_data=["game_id", "spread_line"],
    template="plotly_white",
)

# ==============================================================================
# ANALYSIS 2: Timing & Game Leverage (Flags in Close Games vs. Blowouts)
# ==============================================================================

# Filter for penalty plays only
flags_df = df[df["penalty"] == 1].copy()

# Define absolute score differential at time of penalty
flags_df["abs_score_diff"] = flags_df["score_differential"].abs()

# Categorize game phase by Vegas spread proximity
# (Is the game score differential currently within 1 touchdown of the Vegas spread?)
flags_df["spread_margin"] = (
    flags_df["score_differential_post"] - flags_df["spread_line"]
).abs()

# Group by 5-minute game intervals to check flag frequency vs time remaining
flags_df["time_bucket"] = pd.cut(
    flags_df["game_seconds_remaining"],
    bins=np.arange(0, 3601, 300),
    labels=[f"{i}m remaining" for i in range(5, 61, 5)],
)

time_dist = (
    flags_df.groupby("time_bucket", observed=False)
    .size()
    .reset_index(name="flag_count")
)

fig_time = px.bar(
    time_dist,
    x="time_bucket",
    y="flag_count",
    title="Flag Volume by Game Time Remaining",
    labels={
        "time_bucket": "Time Remaining in Game",
        "flag_count": "Flags Thrown",
    },
    template="plotly_white",
)

# Render interactive plots via HTML to prevent PyCharm GUI warnings
fig_game.write_html("flags_vs_vegas_line.html")
fig_time.write_html("flags_by_time_remaining.html")

print("\nPlots saved to 'flags_vs_vegas_line.html' and 'flags_by_time_remaining.html'")