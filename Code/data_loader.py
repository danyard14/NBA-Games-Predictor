from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
from utils import utils
from collections import defaultdict

data_train_path = '../Data/train_data/17-18_allgames.csv'


###################################### RICH DATA FRAME FUNCTIONS ######################################

def add_number_of_allstar_players(df, all_star_players_path):
    nba_teams = df['Home Team'].values
    df['All Stars Home'] = 0
    df['All Stars Visitor'] = 0
    all_stars_df = pd.read_csv(all_star_players_path)
    team_allstars_dict = dict()
    for nba_team in nba_teams:
        team_allstars_dict[nba_team] = 0

    for index, row in all_stars_df.iterrows():
        team_allstars_dict[utils.symbol_to_team(row['Team'])] += 1

    for index, row in df.iterrows():
        home_team = row['Home Team']
        visitor_team = row['Visitor Team']
        home_allstars = team_allstars_dict[home_team]
        visitor_allstars = team_allstars_dict[visitor_team]
        df.at[index, 'All Stars Home'] = home_allstars
        df.at[index, 'All Stars Visitor'] = visitor_allstars


def add_ranking(df: pd.DataFrame, prev_year_standings_path: str):
    train_standings_path = prev_year_standings_path
    df['Home Rank Higher'] = 0
    standing_f = pd.read_csv(train_standings_path)
    for index, row in df.iterrows():
        home_team = row['Home Team']
        home_index = np.where(standing_f['Team'] == home_team)[0][0]
        home_rank = standing_f.at[home_index, 'Rk']

        visitor_team = row['Visitor Team']
        visitor_index = np.where(standing_f['Team'] == visitor_team)[0][0]
        visitor_rank = standing_f.at[visitor_index, 'Rk']

        if home_rank > visitor_rank:
            df.at[index, 'Home Rank Higher'] = 1
        else:
            df.at[index, 'Home Rank Higher'] = 0


def add_streaks(df: pd.DataFrame):
    # Taking into consideration winning streaks - What are the teams' win streaks coming into the game?
    df["Home Win Streak"] = 0
    df["Visitor Win Streak"] = 0

    from collections import defaultdict
    win_streak = defaultdict(int)

    for index, row in df.iterrows():  # Note that this is not the most efficient method
        home_team = row["Home Team"]
        visitor_team = row["Visitor Team"]
        row["Home Win Streak"] = win_streak[home_team]
        row["Visitor Win Streak"] = win_streak[visitor_team]
        df.loc[index] = row

        # Set current win streak
        if row["Home Win"]:
            win_streak[home_team] += 1
            win_streak[visitor_team] = 0
        else:
            win_streak[home_team] = 0
            win_streak[visitor_team] += 1
    a = 5


def add_home_team_won_last(df):
    who_won_last_match = defaultdict(int)
    df["Home Team Won Last"] = 0
    for index, row in df.iterrows():
        home_team = row["Home Team"]
        visitor_team = row["Visitor Team"]
        matchup = tuple(sorted([home_team, visitor_team]))
        if who_won_last_match[matchup] == row["Home Team"]:
            df.at[index, "Home Team Won Last"] = 1
        else:
            df.at[index, "Home Team Won Last"] = 0
        winner = row["Home Team"] if row["Home Win"] else row["Visitor Team"]
        who_won_last_match[matchup] = winner


def get_data_frame(data_path: str, allstar_path: str, prev_year_standing_path: str):
    df = pd.read_csv(data_path)

    # add winner data
    df['Home Win'] = df['Home Points'] > df['Visitor Points']

    add_ranking(df, prev_year_standing_path)
    add_streaks(df)
    add_home_team_won_last(df)
    add_number_of_allstar_players(df, allstar_path)
    labels = df['Home Win'].values
    return df, labels


def encode_data(data_path: str, allstar_path: str, prev_year_standings_path: str):
    df, labels = get_data_frame(data_path, allstar_path, prev_year_standings_path)
    # x_enhanced = df[['Home Team Won Last', 'All Stars Home', 'All Stars Visitor', 'Home Rank Higher', 'Home Win Streak',
    #                  'Visitor Win Streak']].values
    x_enhanced = df[['Home Team Won Last', 'All Stars Home', 'All Stars Visitor']].values

    encoding = LabelEncoder()
    encoding.fit(df["Home Team"].values)
    home_teams = encoding.transform(df["Home Team"].values)
    visitor_teams = encoding.transform(df["Visitor Team"].values)
    x_teams = np.vstack([home_teams, visitor_teams]).T
    one_hot = OneHotEncoder()
    x_teams = one_hot.fit_transform(x_teams).todense()
    x_all = np.hstack([x_enhanced, x_teams])
    return x_all, labels
