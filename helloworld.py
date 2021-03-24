import pandas as pd
from sklearn.metrics import f1_score
df = pd.read_csv('Data/train_data/17-18_allgames.csv')
df['Home Win'] = df['Home Points'] > df['Visitor Points']
labels = df['Home Win'].values
print(labels)
shape = labels.shape
labels_hometeamwins = [1] * len(labels)
percentage = df['Home Win'].sum() / len(labels)
print(percentage)
print(f1_score(labels,labels_hometeamwins,pos_label=None, average='weighted'))




