import nflreadpy as nfl
import numpy as np
import pandas as pd

# Load data
df = nfl.load_pbp([2024, 2025, 2026]).to_pandas()

# Regex pattern matching the three penalty categories (handles sub-types like Offensive/Defensive Holding)
penalty_pattern = "Roughing the Passer|Pass Interference|Holding"

# Filter for regular season penalties matching target types
flags = df[
    (df["penalty"] == 1)
    & (df["penalty_type"].str.contains(penalty_pattern, case=False, na=False))
    & (df["vegas_wpa"].notna())
    & (df["spread_line"].notna())
].copy()

# Calculate metrics
flags["abs_spread"] = flags["spread_line"].abs()
flags["abs_vegas_wpa"] = flags["vegas_wpa"].abs()

# Directional WPA (positive = favored team helped)
flags["favored_team_wpa"] = flags.apply(
    lambda r: (
        r["vegas_wpa"]
        if r["penalty_team"] != r["home_team"]
        else -r["vegas_wpa"]
    ),
    axis=1,
)

# Categorize spreads
conditions = [
    flags["abs_spread"] <= 3,
    (flags["abs_spread"] > 3) & (flags["abs_spread"] < 10),
    flags["abs_spread"] >= 10,
]
choices = [
    "Close Spread (<= 3 pts)",
    "Moderate Spread (3.5 - 9.5 pts)",
    "Blowout Spread (>= 10 pts)",
]
flags["spread_category"] = np.select(conditions, choices, default="Other")

# Summary Table
summary = (
    flags.groupby("spread_category")
    .agg(
        total_flags=("penalty", "count"),
        mean_abs_wpa=("abs_vegas_wpa", "mean"),
        median_abs_wpa=("abs_vegas_wpa", "median"),
        p90_abs_wpa=("abs_vegas_wpa", lambda x: np.percentile(x, 90)),
        max_abs_wpa=("abs_vegas_wpa", "max"),
        net_favored_wpa=("favored_team_wpa", "mean"),
    )
    .reset_index()
)

print("--- Key Penalties (DPI, Roughing, Holding) WPA Impact ---")
print(summary.to_string(index=False))