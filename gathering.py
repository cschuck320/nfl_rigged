import pandas as pd

BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"


# NFL team names, colors, logo urls, and more.
# Docs: https://nflfastr.com/reference/teams_colors_logos.html
def get_team_colors_logos():
    df = pd.read_parquet(f"{BASE_URL}/teams/teams_colors_logos.parquet")
    df.to_parquet("data/teams_colors_logos.parquet")


# NFL game/schedule data.
# Docs: https://nflreadr.nflverse.com/articles/dictionary_schedules.html
def get_games():
    df = pd.read_parquet(f"{BASE_URL}/schedules/games.parquet")
    df.to_parquet("data/games.parquet")

# Week-level rosters via NFL Shield v2 API, dating back to 2002.
def get_weekly_rosters(year=2026):
    df = pd.read_parquet(f"{BASE_URL}/weekly_rosters/roster_weekly_{year}.parquet")
    df.to_parquet(f"data/weekly_rosters_{year}.parquet")

# Component files for nflverse players build
def get_players_components():
    # OTC
    df_otc = pd.read_parquet(f"{BASE_URL}/players_components/otc_players.parquet")
    df_otc.to_parquet("data/otc_players.parquet")
    # Players
    df_players = pd.read_parquet(f"{BASE_URL}/players_components/players.parquet")
    df_players.to_parquet("data/players_components.parquet")


# Player information for nflverse. This should be the go-to for position and ID mappings going forward.
def get_players():
    df = pd.read_parquet(f"{BASE_URL}/players/players.parquet")
    df.to_parquet("data/players.parquet")


# Participation data for plays from NGS (starts from 2016)
def get_pbp_participation(year=2025):
    df = pd.read_parquet(f"{BASE_URL}/pbp_participation/pbp_participation_{year}.parquet")
    df.to_parquet(f"data/pbp_participation/pbp_participation_{year}.parquet")


# Data about which referees and officials were assigned to specific games.
def get_officials():
    df = pd.read_parquet(f"{BASE_URL}/officials/officials.parquet")
    df.to_parquet("data/officials.parquet")

# Play by play data
# From 1999 to present
def get_play_by_play(year=2026):
    df = pd.read_parquet(f"{BASE_URL}/pbp/play_by_play_{year}.parquet")
    df.to_parquet(f"data/pbp/play_by_play_{year}.parquet")


def print_df_columns(dataframe):
    for col in dataframe.columns:
        print(col)


get_play_by_play()

# # Filter for penalty plays
# penalty_plays = df[df["penalty"] == 1][
#     [
#         "game_id",
#         "desc",
#         "penalty_type",
#         "penalty_team",
#         "penalty_player_name",
#         "penalty_yards",
#     ]
# ]
# print(penalty_plays.head())