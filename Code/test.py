import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

from data_loader import get_data_frame


def test_decision_tree(tree, test_data_path):
    df, labels = get_data_frame(test_data_path)

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
    y_pred = tree.predict(x_all)

    print(accuracy_score(labels, y_pred=y_pred))
