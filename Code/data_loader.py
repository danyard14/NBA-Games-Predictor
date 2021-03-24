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

