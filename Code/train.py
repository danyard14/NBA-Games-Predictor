import pandas as pd
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from test import test_decision_tree
from data_loader import get_data_frame


def train_decision_tree(train_data_path: str):
    df, labels = get_data_frame(train_data_path)

    x_enhanced = df[['Home Team Won Last', 'All Stars Home', 'All Stars Visitor', 'Home Rank Higher', 'Home Win Streak', 'Visitor Win Streak']].values
    encoding = LabelEncoder()
    encoding.fit(df["Home Team"].values)

    home_teams = encoding.transform(df["Home Team"].values)
    visitor_teams = encoding.transform(df["Visitor Team"].values)

    x_teams = np.vstack([home_teams, visitor_teams]).T
    x_all = np.hstack([x_enhanced, x_teams])
    one_hot = OneHotEncoder()
    x_all = one_hot.fit_transform(x_all).todense()

    print(x_all.shape)
    # scorer = make_scorer(f1_score, pos_label=None, average='weighted')
    dtc = DecisionTreeClassifier(random_state=14)
    dtc.fit(x_all, labels)
    return dtc


if __name__ == '__main__':
    t = train_decision_tree('../Data/train_data/17-18_allgames.csv')
    test_decision_tree(t, '../Data/test_data/18_19_allgames.csv')
