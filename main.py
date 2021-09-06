import pandas as pd
from soccer_utils import * 

# https://es.wikipedia.org/wiki/Clasificaci%C3%B3n_de_Conmebol_para_la_Copa_Mundial_de_F%C3%BAtbol_de_2022#Tabla_de_posiciones

N_SIM = 1000
BARS_COLORS_THRESHOLDS = [
    {"max_threshold": 10, "color": "silver", "label": "Not qualified"},
    {"max_threshold": 5, "color": "royalblue", "label": "Inter-confederation play-offs"},
    {"max_threshold": 4, "color": "limegreen", "label": "Qualified"},
]

# Load results to date
parse_raw_matches()
df_matches = pd.read_csv("matches.csv")

# Get teams names
teams = df_matches[LABEL_HOME].unique()

# Build table to date
df_table_to_date = current_table(df_matches, teams)
df_table_to_date.to_csv("df_table_to_date.csv")

print("Simulating remaining games...")
sim_poisson_home, sim_poisson_away = poisson_tournament(df_matches, N_SIM)

print("Summarizing positions...")
df_posicion = summary_positions(
    sim_poisson_home,
    sim_poisson_away, 
    N_SIM, 
    teams, 
    df_table_to_date,
)

print("Exporting probs per team...")
for team in teams:
    print("--- " + team)
    save_team_probs_plot(team, df_posicion, N_SIM, len(teams), BARS_COLORS_THRESHOLDS)