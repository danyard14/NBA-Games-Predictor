from abc import ABC
from torch.utils.data import Dataset, DataLoader
import glob
import pandas
import pandas as pd

data_train_path = '../Data/train_data/17-18_allgames.csv'


class GamesDataset(Dataset, ABC):
    def __init__(self, data_path):
        super(GamesDataset, self).__init__()
        self.data = None
        self.labels = None


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


def get_data_frame(data_path):
    df = pd.read_csv(data_path)

    # add winner data
    df['Home Win'] = df['Home Points'] > df['Visitor Points']

    add_streaks(df)
    # TODO: add:
    #   1. winning streaks (int) [<winning streaks home>, <winning streaks visitor>]
    #   2. amount of all star players for each team (int) [<all star home>, <all star visitor>]
    #   3. home team ranks higher (bool)
    #   4. home team won last time these teams met (bool)
    #   5. home last win, visitor last win [bool, bool]
    #   6. home team usually wins at home (maybe)


    labels = df['Home Win'].values

