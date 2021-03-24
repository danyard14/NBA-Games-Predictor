import pandas as pd
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from test import test_decision_tree

def train_decision_tree(train_data_path: str):
    df = pd.read_csv(train_data_path)

    # add winner data
    df['Home Win'] = df['Home Points'] > df['Visitor Points']
    labels = df['Home Win'].values

    encoding = LabelEncoder()
    encoding.fit(df["Home Team"].values)

    home_teams = encoding.transform(df["Home Team"].values)
    visitor_teams = encoding.transform(df["Visitor Team"].values)

    X_teams = np.vstack([home_teams, visitor_teams]).T

    scorer = make_scorer(f1_score, pos_label=None, average='weighted')
    dtc = DecisionTreeClassifier(random_state=14)

    dtc.fit(X_teams, labels)

    return dtc


if __name__ == '__main__':
    t = train_decision_tree('../Data/train_data/17-18_allgames.csv')
    test_decision_tree(t, '../Data/test_data/18_19_allgames.csv')