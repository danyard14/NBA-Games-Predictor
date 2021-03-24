import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('../Data/train_data/17-18_allgames.csv')

df['Home Win'] = df['Home Points'] > df['Visitor Points']
labels = df['Home Win'].values
# print(labels)
# shape = labels.shape
# labels_hometeamwins = [1] * len(labels)
# percentage = df['Home Win'].sum() / len(labels)
# print(percentage)
# print(f1_score(labels,labels_hometeamwins,pos_label=None, average='weighted'))

encoding = LabelEncoder()
encoding.fit(df["Home Team"].values)

# at game i (home_teams[i]) the home team's number is present.
home_teams = encoding.transform(df["Home Team"].values)
visitor_teams = encoding.transform(df["Visitor Team"].values)

X_teams = np.vstack([home_teams, visitor_teams]).T

# onehot = OneHotEncoder()
# X_teams = onehot.fit_transform(X_teams).todense()

scorer = make_scorer(f1_score, pos_label=None, average='weighted')
dtc = DecisionTreeClassifier(random_state=14)

scores = cross_val_score(dtc, X_teams, labels, scoring=scorer)


dtc.fit(X_teams, labels)
Y_pred = dtc.predict(X_teams)

print(accuracy_score(labels, y_pred=Y_pred))


# Print results
# print("F1: {0:.4f}%".format(np.mean(scores) * 100))
