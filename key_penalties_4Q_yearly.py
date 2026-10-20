import nflreadpy as nfl
import pandas as pd
import plotly.express as px

# 1. Load PBP data
years = list(range(2016, 2027))
df = nfl.load_pbp(years).to_pandas()

# Filter for regular season standard quarters Q1-Q4
df_reg = df[(df["qtr"].isin([1, 2, 3, 4]))].copy()

# Define key judgment penalties (PI, Roughing the Passer, Holding)
penalty_pattern = "Roughing the Passer|Pass Interference|Holding"
df_reg["is_key_penalty"] = (df_reg["penalty"] == 1) & (
    df_reg["penalty_type"].str.contains(penalty_pattern, case=False, na=False)
)

# 2. Group by Season and Quarter
yearly_summary = (
    df_reg.groupby(["season", "qtr"])
    .agg(
        total_plays=("play_id", "count"),
        total_flags=("penalty", "sum"),
        key_flags=("is_key_penalty", "sum"),
    )
    .reset_index()
)

# Normalize rates per 100 plays to control for total snap counts
yearly_summary["all_flags_per_100"] = (
    yearly_summary["total_flags"] / yearly_summary["total_plays"]
) * 100
yearly_summary["key_flags_per_100"] = (
    yearly_summary["key_flags"] / yearly_summary["total_plays"]
) * 100

# 3. Pivot tables for side-by-side comparison
pivot_all = yearly_summary.pivot(
    index="qtr", columns="season", values="all_flags_per_100"
)
pivot_key = yearly_summary.pivot(
    index="qtr", columns="season", values="key_flags_per_100"
)

print("--- ALL PENALTIES: Flags per 100 Plays by Quarter & Year ---")
print(pivot_all.round(2))

print("\n--- KEY PENALTIES (DPI, Roughing, Holding): Flags per 100 Plays ---")
print(pivot_key.round(2))

# Faceted bar charts: One panel per season, placed side-by-side
fig_facet = px.bar(
    yearly_summary,
    x="qtr",
    y="key_flags_per_100",
    color="qtr",
    facet_col="season",
    title="Key Penalty Rate by Quarter — Season Breakdown (2016–2026)",
    labels={
        "qtr": "Quarter",
        "key_flags_per_100": "Key Flags per 100 Plays",
    },
    template="plotly_white",
)

fig_facet.update_xaxes(
    type="category", tickvals=[1, 2, 3, 4], ticktext=["Q1", "Q2", "Q3", "Q4"]
)
fig_facet.write_html("yearly_penalty_facets_historic.html")
print("\nSaved chart to 'yearly_penalty_trends_historic.html'")