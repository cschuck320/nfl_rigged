import nflreadpy as nfl
import numpy as np
import pandas as pd
import plotly.express as px

pd.set_option("display.max_columns", None)

# Define span of time
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