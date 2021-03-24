import pandas as pd
import data_loader
import os

print(os.getcwd())
df = pd.read_csv('../Data/train_data/17-18_allgames.csv')
data_loader.add_number_of_allstar_players(df, '../Data/auxilary_data/17-18_allstars.csv')
print(df.head())
