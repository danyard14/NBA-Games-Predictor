import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np


def test_decision_tree(tree, test_data_path):
    df = pd.read_csv(test_data_path)

    # add winner data
    df['Home Win'] = df['Home Points'] > df['Visitor Points']
    labels = df['Home Win'].values

    encoding = LabelEncoder()
    encoding.fit(df["Home Team"].values)

    home_teams = encoding.transform(df["Home Team"].values)
    visitor_teams = encoding.transform(df["Visitor Team"].values)

    X_teams = np.vstack([home_teams, visitor_teams]).T

    Y_pred = tree.predict(X_teams)

    print(accuracy_score(labels, y_pred=Y_pred))
