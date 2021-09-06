import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import csv

plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (16, 9) 

LABEL_POSITION = "Position"
LABEL_TEAM = "Team"
LABEL_PLAYED = "Played"
LABEL_POINTS = "Points"
LABEL_GF = "GF"
LABEL_GA = "GA"
LABEL_GD = "GD"
LABEL_GOALS_HOME = "GoalsHome"
LABEL_GOALS_AWAY = "GoalsAway"
LABEL_HOME = "Home"
LABEL_AWAY = "Away"
LABEL_RIVAL = "Rival"
LABEL_GOALS = "Goals"
LABEL_IS_HOME_TEAM = "IsHomeTeam"
LABEL_MATCH_ID = "MatchID"

def parse_raw_matches():
    with open("raw_matches.txt", "r") as f:
        lines = [x.strip() for x in f.readlines()]
        lines = [l.split("\t") for l in lines]
    csv_lines = []
    for line in lines:
        goals_home, goals_away = line[4].split(" ")[0].split(":")
        csv_lines.append({
            LABEL_HOME: line[2],
            LABEL_AWAY: line[6],
            LABEL_GOALS_HOME: goals_home,
            LABEL_GOALS_AWAY: goals_away,
            LABEL_PLAYED: goals_home != "" and goals_away != ""
        })
    
    with open('matches.csv', 'w', newline='') as csvfile:
        fieldnames = list(csv_lines[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for line in csv_lines:
            writer.writerow(line)

def current_table(df_matches, teams):
    columns = [LABEL_POSITION, LABEL_TEAM, LABEL_PLAYED, LABEL_POINTS, LABEL_GD, LABEL_GF, LABEL_GA]
    df_table = pd.DataFrame(index = range(len(teams)), columns = columns)

    for t, team in enumerate(teams):
        perf_home = df_matches[(df_matches[LABEL_HOME] == team) & (df_matches[LABEL_PLAYED] == 1)]
        perf_away = df_matches[(df_matches[LABEL_AWAY] == team) & (df_matches[LABEL_PLAYED] == 1)]

        pts_home = len(perf_home[perf_home[LABEL_GOALS_HOME] > perf_home[LABEL_GOALS_AWAY]]) * 3 + len(perf_home[perf_home[LABEL_GOALS_HOME] == perf_home[LABEL_GOALS_AWAY]])
        pts_away = len(perf_away[perf_away[LABEL_GOALS_HOME] < perf_away[LABEL_GOALS_AWAY]]) * 3 + len(perf_away[perf_away[LABEL_GOALS_HOME] == perf_away[LABEL_GOALS_AWAY]])
        pts = pts_home + pts_away

        gf = perf_home[LABEL_GOALS_HOME].sum() + perf_away[LABEL_GOALS_AWAY].sum()
        ga = perf_home[LABEL_GOALS_AWAY].sum() + perf_away[LABEL_GOALS_HOME].sum()
        gd = gf - ga
        played = df_matches[((df_matches[LABEL_HOME] == team) | (df_matches[LABEL_AWAY] == team)) & (df_matches[LABEL_PLAYED] == 1)].shape[0]

        df_table.at[t] = [None, team, played, pts, gd, gf, ga]
     
    df_table.sort_values(
        by=[LABEL_POINTS,LABEL_GD,LABEL_GF,LABEL_GA], 
        inplace = True, 
        ascending = False,
    )
    df_table[LABEL_POSITION] = range(1, len(teams) + 1)
    df_table.set_index(LABEL_TEAM, inplace = True)

    return df_table

def fit_poisson_model(df_matches):
    df_played = df_matches[df_matches[LABEL_PLAYED] == 1]
    
    goals_model_data = pd.concat([
        df_played[[LABEL_HOME, LABEL_AWAY, LABEL_GOALS_HOME]].assign(**{LABEL_IS_HOME_TEAM: 1}).rename(columns={LABEL_HOME: LABEL_TEAM, LABEL_AWAY: LABEL_RIVAL, LABEL_GOALS_HOME: LABEL_GOALS}),
        df_played[[LABEL_AWAY, LABEL_HOME, LABEL_GOALS_AWAY]].assign(**{LABEL_IS_HOME_TEAM: 0}).rename(columns={LABEL_AWAY: LABEL_TEAM, LABEL_HOME: LABEL_RIVAL, LABEL_GOALS_AWAY: LABEL_GOALS}),
    ])

    poisson_model = smf.glm(
        formula="{} ~ {} + {} + {}".format(LABEL_GOALS, LABEL_IS_HOME_TEAM, LABEL_TEAM, LABEL_RIVAL),
        data = goals_model_data, 
        family=sm.families.Poisson(),
    ).fit()

    return poisson_model

def poisson_tournament(df_matches, N):
    poisson_model = fit_poisson_model(df_matches)

    df_non_played_matches = df_matches[df_matches[LABEL_PLAYED] == 0]

    index = range(df_non_played_matches.shape[0])
    columns = range(1, N + 1)

    sim_poisson_home = pd.DataFrame(index=index, columns=columns)
    sim_poisson_away = pd.DataFrame(index=index, columns=columns)
    sim_poisson_home[LABEL_MATCH_ID] = df_non_played_matches.index
    sim_poisson_home[LABEL_HOME] = df_non_played_matches[LABEL_HOME].values
    sim_poisson_away[LABEL_MATCH_ID] = df_non_played_matches.index
    sim_poisson_away[LABEL_AWAY] = df_non_played_matches[LABEL_AWAY].values

    sim_poisson_home.set_index([LABEL_MATCH_ID, LABEL_HOME], inplace = True)
    sim_poisson_away.set_index([LABEL_MATCH_ID, LABEL_AWAY], inplace = True)

    for i, _ in tqdm(enumerate(df_non_played_matches.index)):
        home = df_non_played_matches.iloc[i][LABEL_HOME]
        away = df_non_played_matches.iloc[i][LABEL_AWAY]
        lambda_home = poisson_model.predict(pd.DataFrame(data={LABEL_TEAM: home, LABEL_RIVAL: away, LABEL_IS_HOME_TEAM: 1}, index = [1]))
        lambda_away = poisson_model.predict(pd.DataFrame(data={LABEL_TEAM: away, LABEL_RIVAL: home, LABEL_IS_HOME_TEAM: 0}, index = [1]))
        goals_home = np.random.poisson(lambda_home, N)
        goals_away = np.random.poisson(lambda_away, N)

        sim_poisson_home.iloc[i] = goals_home
        sim_poisson_away.iloc[i] = goals_away

    return sim_poisson_home, sim_poisson_away

def summary_positions(sim_poisson_home, sim_poisson_away, N_SIM, teams, df_table_to_date):
    a_dict = {}
    b_dict = {}
    aa_dict = {}
    bb_dict = {}

    for team in teams:
        # info GF
        a = sim_poisson_home[sim_poisson_home.index.get_level_values(LABEL_HOME) == team].reset_index().drop(LABEL_HOME, axis = 1).set_index(LABEL_MATCH_ID)
        # info GA
        b = sim_poisson_away[sim_poisson_away.index.get_level_values(LABEL_AWAY) == team].reset_index().drop(LABEL_AWAY, axis = 1).set_index(LABEL_MATCH_ID)
        # info away teams
        aa = sim_poisson_away[sim_poisson_away.index.get_level_values(0).isin(a.index.get_level_values(LABEL_MATCH_ID))].reset_index().drop(LABEL_AWAY, axis = 1).set_index(LABEL_MATCH_ID)
        bb = sim_poisson_home[sim_poisson_home.index.get_level_values(0).isin(b.index.get_level_values(LABEL_MATCH_ID))].reset_index().drop(LABEL_HOME, axis = 1).set_index(LABEL_MATCH_ID)

        a_dict[team] = a
        b_dict[team] = b
        aa_dict[team] = aa
        bb_dict[team] = bb

    # team, n_sim, position
    team_stats = []
    for n_sim in tqdm(range(1, N_SIM + 1)):
        df_table_sim = df_table_to_date.copy()
        for team in teams:
            # info home teams
            a = a_dict[team]
            b = b_dict[team]
            # info away teams
            aa = aa_dict[team]
            bb = bb_dict[team]
            pts = 3 * (sum(a[n_sim] > aa[n_sim])) + (sum(a[n_sim] == aa[n_sim])) + 3 * (sum(b[n_sim] > bb[n_sim])) + (sum(b[n_sim] == b[n_sim]))
            gf = sum(a[n_sim])
            gc = sum(b[n_sim])
            df_table_sim.loc[team, LABEL_POINTS] += pts
            df_table_sim.loc[team, LABEL_GF] += gf
            df_table_sim.loc[team, LABEL_GA] += gc
        df_table_sim[LABEL_GD] = df_table_sim[LABEL_GF] - df_table_sim[LABEL_GA]
            
        df_table_sim.sort_values(by=[LABEL_POINTS,LABEL_GD,LABEL_GF,LABEL_GA], inplace = True, ascending = [False, False, False, True])
        df_table_sim[LABEL_POSITION] = range(1, len(teams) + 1)
            
        for team in teams:
            team_stats.append([team, "Absoluta", n_sim, df_table_sim.loc[team, LABEL_POSITION]])

    df_posicion = pd.DataFrame(team_stats, columns = [LABEL_TEAM, "Tabla", "n_sim", LABEL_POSITION])
    return df_posicion

def save_team_probs_plot(team, df_posicion, N_sim, N_teams, bars_colors_thresholds):
    plt.clf()

    p_pos_abs = df_posicion[(df_posicion.Tabla == "Absoluta") &
        (df_posicion[LABEL_TEAM] == team)][LABEL_POSITION].value_counts().sort_index(ascending = False) * 100 / N_sim

    bars_colors = []
    for i in p_pos_abs.index:
        color = "silver"
        for threshold in bars_colors_thresholds:
            if i <= threshold["max_threshold"]:
                color = threshold["color"]
        bars_colors.append(color)

    plt.bar(p_pos_abs.index, p_pos_abs.values, color = bars_colors)

    for index in p_pos_abs.index:
        value = p_pos_abs[index]
        if value > 1 and value < 99:
            plt.text(x = index - 0.4, y = value + 0.5, fontsize = 16, s = str(int(round(value))) + "%")
        else:
            plt.text(x = index - 0.4, y = value + 0.5, fontsize = 16, s = str(round(value, 1)) + "%")

    plt.title(team, fontsize = 30)

    bars_colors_thresholds = bars_colors_thresholds[::-1]
    plt.legend(
        [Line2D([0], [0], color = threshold["color"], lw = 4) for threshold in bars_colors_thresholds], 
        [threshold["label"] for threshold in bars_colors_thresholds], 
        fontsize = 16,
    )

    plt.xlabel(LABEL_POSITION, fontsize = 16)
    plt.ylabel("Odds of ending in position", fontsize = 16)

    plt.xticks(np.arange(0, N_teams + 1), fontsize = 16)
    plt.yticks(fontsize = 16)

    plt.xlim(0.01, N_teams + 1)
    plt.ylim(0, max(p_pos_abs) * 1.1)

    plt.savefig('./plots/{}.png'.format(team), bbox_inches='tight')

    plt.clf()
