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

PBP_COLS = [
    "play_id",  # Numeric play id that when used with game_id and drive provides the unique identifier for a single play.
    "game_id",
    "old_game_id",
    "home_team",
    "away_team",
    "season_type",
    "week",
    "posteam",  # String abbreviation for the team with possession.
    "posteam_type",  # String indicating whether the posteam team is home or away.
    "quarter_seconds_remaining",
    "game_seconds_remaining",
    "sp",  # Binary indicator for whether or not a score occurred on the play.
    "qtr",
    "down",
    "goal_to_go",  # Binary indicator for whether or not the posteam is in a goal down situation.
    "time",
    "yrdln",  # String indicating the current field position for a given play.
    "ydstogo",  # Numeric yards in distance from either the first down marker or the endzone in goal down situations.
    "play_type",
    "yards_gained",
    "total_home_score",  # Score for the home team at the start of the play
    "total_away_score",  # Score for the away team at the start of the play
    "score_differential",  # Score differential between the posteam and defteam at the start of the play
    "td_prob",  # Predicted probability of the posteam scoring a TD next. 'Next' in this context means the next score in the same game half.
    "ep",  # Using the scoring event probabilities, the estimated expected points with respect to the possession team for the given play.
    "epa",  # Expected points added (EPA) by the posteam for the given play
    "wp",  # Estimated win probabiity for the posteam given the current situation at the start of the given play
    "def_wp",  # Estimated win probability for the defteam.
    "wpa",  # Win probability added (WPA) for the posteam
    "vegas_wpa",  # Win probability added (WPA) for the posteam: spread_adjusted model
    "vegas_home_wpa",  # Win probability added (WPA) for the home team: spread_adjusted model.
    "home_wp_post",  # Estimated win probability for the home team at the end of the play.
    "away_wp_post",  # Estimated win probability for the away team at the end of the play.
    "vegas_wp",  # Estimated win probabiity for the posteam given the current situation at the start of the given play, incorporating pre-game Vegas line.
    "vegas_home_wp",  # Estimated win probability for the home team incorporating pre-game Vegas line.
    "first_down_penalty",  # Binary indicator for if a penalty converted the first down.
    "third_down_converted",  # Binary indicator for if the first down was converted on third down.
    "third_down_failed",  # Binary indicator for if the posteam failed to convert first down on third down.
    "fourth_down_converted",  # Binary indicator for if the first down was converted on fourth down.
    "fourth_down_failed",  # Binary indicator for if the posteam failed to convert first down on fourth down.
    "penalty_team",  # String abbreviation of the team with the penalty.
    "penalty_player_id",  # Unique identifier for the player with the penalty.
    "penalty_player_name",  # String name for the player with the penalty.
    "penalty_yards",  # Yards gained (or lost) by the posteam from the penalty.
    "penalty_type",  # String indicating the penalty type of the first penalty in the given play. Will be `NA` if `desc` is missing the type.
    "season",  # 4 digit number indicating to which season the game belongs to.
    "spread_line",  # The closing spread line for the game. A positive number means the home team was favored by that many points, a negative number means the away team was favored by that many points
    "total_line",  # The closing total line for the game
    "div_game",  # Binary indicator for if the given game was a division game.
    "first_down",  # Binary indicator if the play ended in a first down.
    "aborted_play",  # Binary indicator if the play description indicates "Aborted".
]


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
    """
    Aggregate game info with schedules and officials.

    :param int start_date: Starting date for data.
    :param int end_date: Ending date (+1 is automatically to include passed year).
    :return pd.Pandas: Pandas DataFrame with aggregated game info.
    """

    schedules = get_schedules(start_date, end_date)
    officials = get_officials()
    officials = officials.rename(columns={"game_id": "old_game_id"})
    return pd.merge(officials, schedules, on="old_game_id")


def agg_game_info_pbp(start_date, end_date):
    """
    Aggregates game info with schedules and officials and added play-by-play.

    :param int start_date: Starting date.
    :param int end_date: Ending date (+1 is automatically to include passed year).
    :return pd.Pandas: Pandas DataFrame with aggregated play-by-play and game info.:
    """
    merge_cols = [
        "game_id", "old_game_id", "season", "week", "away_team", "away_score", "home_team", "home_score",
        "location", "result", "total", "spread_line", "total_line", "div_game", "roof", "surface", "temp",
        "wind", "away_coach", "home_coach", "stadium_id", "stadium"
    ]
    return pd.merge(agg_game_info(start_date, end_date), get_pbp(start_date, end_date), on=merge_cols)


def filter_columns(df, columns):
    return df[columns]


def print_df_columns(dataframe):
    # For finding the columns in a query.
    for col in dataframe.columns:
        print(col)

