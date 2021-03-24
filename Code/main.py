import pandas as pd
import data_loader
import os

print(os.getcwd())
df = data_loader.get_data_frame('../Data/train_data/17-18_allgames.csv')
data_loader.add_home_team_won_last(df)
print(df.head(250))
