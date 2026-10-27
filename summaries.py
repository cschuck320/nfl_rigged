import pandas as pd
import numpy as np
import gathering as gat
import plotly.express as px



def penalties_per_quarter_yearly(df, start_year, end_year):
    # Filter for regular season quarters Q1-Q4
    df_reg = df[(df["game_type"] == "REG") & (df["qtr"].isin([1, 2, 3, 4]))].copy()

    # Define key judgment penalties (PI, Roughing the Passer, Holding)
    penalty_pattern = "Roughing the Passer|Pass Interference|Holding"
    df_reg["is_key_penalty"] = (df_reg["penalty"] == 1) & (
        df_reg["penalty_type"].str.contains(penalty_pattern, case=False, na=False)
    )

    # Group by Season and Quarter
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
    yearly_summary["all_flags_per_100"] = (yearly_summary["total_flags"] / yearly_summary["total_plays"]) * 100
    yearly_summary["key_flags_per_100"] = (yearly_summary["key_flags"] / yearly_summary["total_plays"]) * 100

    # Pivot tables for comparison between all flags and key flags
    pivot_all = yearly_summary.pivot(index="qtr", columns="season", values="all_flags_per_100")
    pivot_key = yearly_summary.pivot(index="qtr", columns="season", values="key_flags_per_100")

    # Reshape summary into long format to support overlaid bars
    summary_long = yearly_summary.melt(
        id_vars=["season", "qtr"],
        value_vars=["all_flags_per_100", "key_flags_per_100"],
        var_name="penalty_type",
        value_name="flags_per_100",
    )
    summary_long["penalty_type"] = summary_long["penalty_type"].map(
        {
            "all_flags_per_100": "All Penalties",
            "key_flags_per_100": "Key Penalties",
        }
    )

    # Faceted overlaid bar chart: "All Penalties" rendered behind "Key Penalties"
    fig_facet = px.bar(
        summary_long,
        x="qtr",
        y="flags_per_100",
        color="penalty_type",
        facet_col="season",
        barmode="overlay",
        category_orders={"penalty_type": ["All Penalties", "Key Penalties"]},
        color_discrete_map={
            "All Penalties": "#CBD5E1",  # Light gray background bar
            "Key Penalties": "#013369",  # NFL dark blue foreground bar
        },
        title=f"Penalty Rates by Quarter — Season Breakdown ({start_year}–{end_year})",
        labels={
            "qtr": "Quarter",
            "flags_per_100": "Flags per 100 Plays",
            "penalty_type": "Penalty Type",
        },
        template="plotly_white",
    )

    # Heatmap plot for pivot_all (Flags / 100 plays)
    fig_all = px.imshow(
        pivot_all.round(2),
        labels=dict(x="Season", y="Quarter", color="Flags/100 Plays"),
        x=[str(c) for c in pivot_all.columns],
        y=["Q1", "Q2", "Q3", "Q4"],
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title=f"All Penalties per 100 Plays Matrix ({start_year}–{end_year})",
        template="plotly_white",
    )

    # Heatmap plot for pivot_key (Key Flags / 100 plays)
    fig_key = px.imshow(
        pivot_key.round(2),
        labels=dict(x="Season", y="Quarter", color="Key Flags/100 Plays"),
        x=[str(c) for c in pivot_key.columns],
        y=["Q1", "Q2", "Q3", "Q4"],
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Reds",
        title=f"Key Penalties (PI, Roughing, Holding) per 100 Plays Matrix ({start_year}–{end_year})",
        template="plotly_white",
    )

    return fig_facet, fig_all, fig_key


def flags_thrown_per_quarter_trend(df, start_year, end_year):
    year_range_str = f"{start_year}-{end_year}" if start_year != end_year else f"{start_year}"
    # Calculate total unique games played per season (denominator)
    games_per_season = df.groupby("season")["game_id"].nunique().reset_index(name="total_games")
    # Filter flags within quarters 1-5 and season range
    df_flags = df[(df["penalty"] == 1) & (df["qtr"].isin([1, 2, 3, 4]))].copy()

    # Aggregate counts by season and quarter
    df_counts = (
        df_flags.groupby(["season", "qtr"])
        .size()
        .reset_index(name="flag_count")
    )

    # Merge game totals and compute normalized average per game
    df_counts = df_counts.merge(games_per_season, on="season")
    df_counts["flags_per_game"] = df_counts["flag_count"] / df_counts["total_games"]

    # Plot normalized trend
    fig = px.line(
        df_counts,
        x="season",
        y="flags_per_game",
        color="qtr",
        markers=True,
        title=f"Average NFL Flags Thrown per Game by Quarter ({year_range_str})",
        labels={
            "season": "Season",
            "flags_per_game": "Avg Flags / Game",
            "qtr": "Quarter",
        },
    )

    fig.update_xaxes(dtick=1)
    fig.update_layout(
        xaxis_title="Season",
        yaxis_title="Average Flags per Game",
        template="plotly_white",
        hovermode="x unified",
    )

    # Render figure
    fig.write_html(f"nfl_flags_per_game_trend_{year_range_str}.html")
    return fig
