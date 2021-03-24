from abc import ABC
import glob
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


def add_home_team_won_last(df):
    who_won_last_match = defaultdict(int)
    def



    df["Home Team Won Last"] = 0
    for index, row in df.iterrows():
        home_team = row['Home Team']
        visitor_team = row['Visistor Team']

        # Sort for a consistent ordering
        teams = tuple(sorted([home_team, visitor_team]))
        # Parse the row for which team won the last matchup, then add a 1 if the Home Team won
        result = 1 if last_game_winner[teams] == row['Home Team'] else 0

        # Update record for next matchup
        winner = row['Home Team'] if row['Home Win'] else row['Visitor Team']
        last_game_winner[teams] = winner

















def get_data_frame(data_path):
    df = pd.read_csv(data_path)

    # add winner data
    df['Home Win'] = df['Home Points'] > df['Visitor Points']

    # TODO: add:
    #   1. winning streaks (int) [<winning strikes home>, <winning strikes visitor>]
    #   2. amount of all star players for each team (int) [<all star home>, <all star visitor>]
    #   3. home team ranks higher (bool)
    #   4. home team won last time these teams met (bool)
    #   5. home last win, visitor last win [bool, bool]
    #   6. home team usually wins at home (maybe)

    labels = df['Home Win'].values
