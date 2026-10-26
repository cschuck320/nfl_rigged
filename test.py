import nflreadpy as nfl
import numpy as np
import pandas as pd
import plotly.express as px

pd.set_option("display.max_columns", None)

# Define span of time
years = list(range(2020, 2027))
# Get referee data
ref_df = nfl.load_officials(years).to_pandas()
print(ref_df.head(10))