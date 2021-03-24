from abc import ABC
import glob
import pandas
import pandas as pd

data_train_path = '../Data/train_data/17-18_allgames.csv'


###################################### RICH DATA FRAME FUNCTIONS ######################################

def get_number_of_allstar_players(df, all_star_players_path):
    nba_teams = df['Home Team'].values
    all_stars_df = pd.read_csv(all_star_players_path)
    nba_teams_symbols = all_stars_df['Team'].values
    team_allstars_dict = dict()
    for



def get_data_frame(data_path):
    df = pd.read_csv(data_path)

    # add winner data
    df['Home Win'] = df['Home Points'] > df['Visitor Points']

    # TODO: add:
    #   1. winning strikes (int) [<winning strikes home>, <winning strikes visitor>]
    #   2. amount of all star players for each team (int) [<all star home>, <all star visitor>]
    #   3. home team ranks higher (bool)
    #   4. home team won last time these teams met (bool)
    #   5. home last win, visitor last win [bool, bool]
    #   6. home team usually wins at home (maybe)

    labels = df['Home Win'].values
