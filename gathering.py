import pandas as pd
import plotly.express as px
import plotly.io as pio

# Choose the renderer that matches your environment:
pio.renderers.default = (
    "browser"  # Opens in your default system browser (Firefox/Chrome)
)
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


# Data about which referees and officials were assigned to specific games.
def get_officials():
    df = pd.read_parquet(f"{BASE_URL}/officials/officials.parquet")
    df.to_parquet("data/officials.parquet")

# Play by play data
# From 1999 to present
def get_pbp(year=2026):
    df = pd.read_parquet(f"{BASE_URL}/pbp/play_by_play_{year}.parquet")
    df.to_parquet(f"data/pbp/play_by_play_{year}.parquet")


# Participation data for plays from NGS (starts from 2016)
def get_pbp_participation(year=2026):
    print(f"Getting play-by-play participation for {year}...")
    df = pd.read_parquet(f"{BASE_URL}/pbp_participation/pbp_participation_{year}.parquet")
    df.to_parquet(f"data/pbp_participation/pbp_participation_{year}.parquet")
    print(f"{year} play-by-play participation saved to parquet in data/pbp_participation")


# Injuries data (2009-present)
def get_injuries(year=2026):
    print(f"Getting injuries for {year}...")
    df = pd.read_parquet(f"{BASE_URL}/injuries/injuries_{year}.parquet")
    df.to_parquet(f"data/injuries/injuries_{year}.parquet")
    print(f"{year} injuries saved to parquet in data/injuries")


def print_df_columns(dataframe):
    for col in dataframe.columns:
        print(col)

data = pd.read_parquet("data/pbp/play_by_play_2025.parquet")
flags_data = data[data["penalty"] == 1].copy()
flags_data = flags_data[flags_data["qtr"].isin([1, 2, 3, 4, 5])]
flags_data["qtr"] = flags_data["qtr"].astype("int")

fig = px.histogram(
    flags_data,
    x="qtr",
    title="Total NFL Flags Thrown by Quarter (2025)",
    labels={"qtr": "Quarter", "count": "Flags Thrown"},
    color_discrete_sequence=["#013369"],  # NFL Blue
)

# 4. Format axes and layout
fig.update_xaxes(
    type="category",
    tickvals=[1, 2, 3, 4, 5],
    ticktext=["Q1", "Q2", "Q3", "Q4", "OT"],
)

fig.update_layout(
    xaxis_title="Quarter",
    yaxis_title="Number of Flags",
    bargap=0.15,
    template="plotly_white",
)

# Render figure
fig.write_image("nfl_flags_2025.png")

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