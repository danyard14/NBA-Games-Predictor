from abc import ABC
import glob
import numpy as np
import pandas
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

def add_ranking(df: pd.DataFrame):
    train_standings_path = '../Data/auxilary_data/17-18_standings.csv'
    df['Home Rank Higher'] = False
    standing_f = pd.read_csv(train_standings_path)
    for index, row in df.iterrows():
        home_team = row['Home Team']
        home_index = np.where(standing_f['Team'] == home_team)[0][0]
        home_rank = standing_f.at[home_index, 'Rk']

        visitor_team = row['Visitor Team']
        visitor_index = np.where(standing_f['Team'] == visitor_team)[0][0]
        visitor_rank = standing_f.at[visitor_index, 'Rk']

        df.at[index, 'Home Rank Higher'] = home_rank > visitor_rank


def add_streaks(df: pd.DataFrame):
    # Taking into consideration winning streaks - What are the teams' win streaks coming into the game?
    df["Home Win Streak"] = 0
    df["Visitor Win Streak"] = 0

    # Did the home and visitor teams win their last game?
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


def get_data_frame(data_path):
    df = pd.read_csv(data_path)

    # add winner data
    df['Home Win'] = df['Home Points'] > df['Visitor Points']

    add_ranking(df)
    add_streaks(df)
    add_home_team_won_last(df)
    add_number_of_allstar_players(df, '../Data/auxilary_data/17-18_allstars.csv')

    # TODO: add:
    #   1. winning streaks (int) [<winning streaks home>, <winning streaks visitor>]
    #   1. winning streaks (int) [<winning strikes home>, <winning strikes visitor>]
    #   2. amount of all star players for each team (int) [<all star home>, <all star visitor>]
    #   3. home team ranks higher (bool)
    #   4. home team won last time these teams met (bool)
    #   5. home last win, visitor last win [bool, bool]

    #   6. home team usually wins at home (maybe)

    labels = df['Home Win'].values
    return df, labels


if __name__ == '__main__':
    get_data_frame('../Data/train_data/17-18_allgames.csv')