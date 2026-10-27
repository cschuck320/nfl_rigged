import gathering as gat
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
import time

REFEREE_COLS = ["Back Judge", "Down Judge", "Field Judge", "Line Judge", "Referee", "Side Judge", "Umpire"]

TARGET_COLS = ["home_cover", "over_hit", "underdog_outright_win"]

IMPORTANT_COLS = [
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "total_line",
    "under_odds",
    "over_odds",
    "wpa",
    "vegas_wpa",
    "vegas_home_wpa",
    "home_wp_post",
    "away_wp_post",
    "vegas_wp",
    "vegas_home_wp"
]


def create_target_variables(df):
    df_copy = df.copy()
    # If the home team is favored by 3, the line is 3.0.
    # They cover if home_margin + spread_line > 0.
    # Example: Home wins by 4 (home_margin = 4), spread is -3. 4 + (-3) = 1 (Cover = 1).
    # Note: This logic assigns a "0" to pushes (ties against the spread).
    df_copy["home_cover"] = np.where(df_copy["result"] > df_copy["spread_line"], 1, 0)

    # Over Hit (Binary: 1 if Total Points > Total Line, 0 otherwise)
    df_copy["over_hit"] = np.where(df_copy["total"] > df_copy["total_line"], 1, 0)

    # The team with the higher (usually positive) ML is the underdog.
    df_copy["home_is_underdog"] = df_copy["home_moneyline"] > df_copy["away_moneyline"]
    df_copy["away_is_underdog"] = df_copy["away_moneyline"] > df_copy["home_moneyline"]

    # Determine if an underdog won the game outright using the "result" column
    # result > 0 means the Home team won, result < 0 means the Away team won
    df_copy["underdog_outright_win"] = np.where(
        (df_copy["home_is_underdog"] & (df_copy["result"] > 0)) |
        (df_copy["away_is_underdog"] & (df_copy["result"] < 0)),
        1,
        0
    )

    return df_copy


def chi2_test():
    agg_df = gat.agg_game_info(2025, 2026)
    print(f"Length of agg_df: {len(agg_df)}")
    df = create_target_variables(agg_df)
    df_games = df.drop_duplicates(subset=['game_id']).copy()
    print(f"Length of df_games: {len(df_games)}")
    individual_results = []
    # Loop through each target variable
    for target in TARGET_COLS:
        for pos in REFEREE_COLS:
            if pos in df_games.columns and target in df_games.columns:
                # Get a list of all unique officials in this position
                officials = df_games[df_games[pos].notna()][pos].unique()
                for official in officials:
                    if official == 'Unknown':  # Skip our placeholder
                        continue
                    # Create a binary column: 1 if this official worked the game, 0 if they didn't
                    is_official = (df_games[pos] == official).astype(int)
                    # Build a 2x2 contingency table (Official vs The Field)
                    contingency_table = pd.crosstab(is_official, df_games[target])
                    # Ensure the table is 2x2 (meaning the ref has both wins and losses in the target)
                    if contingency_table.shape == (2, 2):
                        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                        # Calculate their specific hit rate
                        games_worked = is_official.sum()
                        target_hits = df_games[(is_official == 1) & (df_games[target] == 1)].shape[0]
                        # Avoid division by zero
                        hit_rate = target_hits / games_worked if games_worked > 0 else 0
                        # Set a minimum game threshold (e.g., 30 games) to remove noise
                        # if games_worked >= 30:
                        individual_results.append({
                            "Target Variable": target,
                            "Position": pos,
                            "Official Name": official,
                            "Games Worked": games_worked,
                            "Hit Rate": round(hit_rate, 3),
                            "P-Value": p_value,
                            "Chi-Square Stat": chi2_stat
                        })
    # Convert the results to a DataFrame
    individual_results_df = pd.DataFrame(individual_results)
    # Sort by P-Value to see the most statistically significant individual referees at the top
    significant_individuals = individual_results_df.sort_values(by="P-Value")
    # Display the top 20 most significant individual trends
    print(significant_individuals.head(20).to_string(index=False))
    # Calculate the total number of valid tests run
    total_tests = len(individual_results_df)
    # Apply Bonferroni Correction to find the true alpha
    adjusted_alpha = 0.05 / total_tests
    # Filter your results by the adjusted threshold
    true_significant_refs = individual_results_df[individual_results_df['P-Value'] <= adjusted_alpha]
    print(f"Total Tests Run: {total_tests}")
    print(f"Adjusted P-Value Threshold: {adjusted_alpha:.6f}")
    print(true_significant_refs)
    time.sleep(1)


def ttest_test():
    df = gat.agg_game_info_pbp(2024, 2026)
    # Filter regular season plays with valid score differentials and spreads
    df_reg = df[(df["game_type"] == "REG") & df["qtr"].isin([1, 2, 3, 4])].copy()

    # Key Judgment Penalties (PI, Roughing, Holding)
    penalty_pattern = "Roughing the Passer|Pass Interference|Holding"
    # penalty_pattern = "Roughing the Passer"
    df_reg["is_key_penalty"] = (df_reg["penalty"] == 1) & (
        df_reg["penalty_type"].str.contains(penalty_pattern, case=False, na=False)
    )

    # Isolate the plays that actually matter
    # Only want plays with a key penalty that actually shifted Win Probability
    high_leverage_penalties = df_reg[
        (df_reg["is_key_penalty"] == True) &
        (df_reg["vegas_home_wpa"].notna()) &
        (df_reg["qtr"] == 4)
    ].copy()

    # Setup for the T-Test
    # Focus on the Head Referee since these fucks have the most discretionary power
    # over Roughing the Passer and often announce the final calls.
    target_wpa = "vegas_home_wpa"
    position = "Referee"

    wpa_results = []
    referees = high_leverage_penalties[position].dropna().unique()

    for ref in referees:
        if ref == 'Unknown':
            continue

        # Isolate the WPA swings from penalties called by THIS referee
        ref_wpa = high_leverage_penalties[high_leverage_penalties[position] == ref][target_wpa]

        # Isolate the WPA swings from penalties called by EVERYONE ELSE (The Field)
        field_wpa = high_leverage_penalties[high_leverage_penalties[position] != ref][target_wpa]

        # Ensure a minimum sample size of key penalties to avoid noise
        if len(ref_wpa) >= 50:
            # Run a Two-Sample T-Test
            t_stat, p_value = ttest_ind(ref_wpa, field_wpa, equal_var=False)

            wpa_results.append({
                "Official Name": ref,
                "Key Penalties Called": len(ref_wpa),
                "Ref Avg Home WPA": round(ref_wpa.mean(), 4),
                "Field Avg Home WPA": round(field_wpa.mean(), 4),
                "T-Statistic": round(t_stat, 3),
                "P-Value": p_value
            })

    # Format and output the results
    wpa_df = pd.DataFrame(wpa_results)

    # Apply Bonferroni correction for the number of Referees tested
    wpa_df['Significant (Bonferroni)'] = wpa_df['P-Value'] < (0.05 / len(wpa_df))

    # Sort by the largest absolute T-Statistic (most extreme divergence from the average)
    wpa_df['Abs_T'] = wpa_df['T-Statistic'].abs()
    significant_wpa = wpa_df.sort_values(by="Abs_T", ascending=False).drop(columns=['Abs_T'])

    print(significant_wpa.head(15).to_string(index=False))


# chi2_test()
ttest_test()
