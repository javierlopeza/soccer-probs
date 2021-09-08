import pandas as pd
from soccer_utils import * 

N_SIM = 10000
TOURNAMENT = "Chile" # Eliminatorias | Chile

# -------------------
# RAW MATCHES SOURCES
# -------------------

# Eliminatorias Qatar 2022
#   https://es.wikipedia.org/wiki/Clasificaci%C3%B3n_de_Conmebol_para_la_Copa_Mundial_de_F%C3%BAtbol_de_2022#Resultados

# Campeonato Chileno
#   https://chile.as.com/resultados/futbol/chile/calendario/

# ----------------------
# BARS COLORS THRESHOLDS
# ----------------------

# Eliminatorias Qatar 2022 
BARS_COLORS_THRESHOLDS = [
    {"max_threshold": 10, "color": "silver", "label": "Not qualified"},
    {"max_threshold": 5, "color": "royalblue", "label": "Inter-confederation play-offs"},
    {"max_threshold": 4, "color": "limegreen", "label": "Qualified"},
]

# Campeonato Chileno
# TODO

# ----------------------

# Load results to date
if TOURNAMENT == "Eliminatorias":
    parse_raw_matches_eliminatorias_qatar_2022()
if TOURNAMENT == "Chile":
    parse_raw_matches_campeonato_chileno()
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
    save_team_probs_plot(TOURNAMENT, team, df_posicion, N_SIM, len(teams), BARS_COLORS_THRESHOLDS)