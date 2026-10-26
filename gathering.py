import pandas as pd
import numpy as np
import plotly.io as pio
import nflreadpy as nfl
from nflreadpy.config import update_config

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
update_config(
    cache_mode="memory",
    cache_dir="~/.cache/nflreadpy",
    cache_duration=86400,
    verbose=False,
    timeout=30,
    user_agent="nflreadpy/v0.1.1"
)

def get_players():
    print("Getting player data...")
    return nfl.load_players().to_pandas()


# Data about which referees and officials were assigned to specific games.
def get_officials():
    # Get referee data
    df = nfl.load_officials().to_pandas()
    pivot_df = df.pivot(index="game_id", columns="position", values="official_name")
    # Drops any column where "Alternate" appears in the column name or MultiIndex tuple
    pivot_df = (pivot_df
                .drop(columns=[col for col in pivot_df.columns if "Alternate" in str(col)])
                .reset_index()
                .rename_axis(columns=None)
                .sort_values(by=["game_id"], ascending=True)
                )
    pivot_df = pivot_df.drop(columns=["AL", "RA", "Replay Official"])
    # Fill missing Down Judge values with Head Linesman values
    pivot_df["Down Judge"] = pivot_df["Down Judge"].fillna(pivot_df["Head Linesman"])
    # Drop the now-redundant Head Linesman column
    pivot_df = pivot_df.drop(columns=["Head Linesman"])
    # Fill remaining negligible nulls with a string placeholder
    minor_missing_cols = ["Field Judge", "Line Judge", "Umpire", "Down Judge"]
    pivot_df[minor_missing_cols] = pivot_df[minor_missing_cols].fillna("Unknown")
    return pivot_df


# Play by play data from 1999 to present
def get_pbp(start_date, end_date):
    print(f"Getting play-by-play data from {start_date} to {end_date}...")
    years = list(range(start_date, end_date+1))
    return nfl.load_pbp(years).to_pandas()


# Participation data for plays from NGS (starts from 2016)
def get_pbp_participation(start_date, end_date):
    print(f"Getting play-by-play participation from {start_date} to {end_date}...")
    years = list(range(start_date, end_date + 1))
    return nfl.load_participation(years).to_pandas()


def get_schedules(start_date, end_date):
    print(f"Getting schedules from {start_date} to {end_date}...")
    years = list(range(start_date, end_date+1))
    return nfl.load_schedules(years).to_pandas()


def agg_game_info(start_date, end_date):
    # Aggregate game info including each ref and the money lines
    schedules = get_schedules(start_date, end_date)
    officials = get_officials()
    officials = officials.rename(columns={"game_id": "old_game_id"})
    return pd.merge(officials, schedules, on="old_game_id")


def agg_game_info_pbp(start_date, end_date):
    merge_cols = ["game_id", "old_game_id", "season", "week", "away_team", "away_score", "home_team", "home_score", "location", "result", "total", "spread_line", "total_line", "div_game", "roof", "surface", "temp", "wind", "away_coach", "home_coach", "stadium_id", "stadium"]
    return pd.merge(agg_game_info(start_date, end_date), get_pbp(start_date, end_date), on=merge_cols)


def print_df_columns(dataframe):
    # For finding the columns in a query.
    for col in dataframe.columns:
        print(col)

