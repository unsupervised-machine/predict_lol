# Script to grab the most recent copy of the data, also stores a copy for a historic view

import pandas as pd
import time

timestr = time.strftime("%Y-%m-%d")
url = ('https://raw.githubusercontent.com/ds-leehanjin/league-of-legends-outcome-classification/master/data/high_diamond_ranked_10min.csv')
df = pd.read_csv(url)
df.to_csv('data\league_games.csv')
df.to_csv('data\historical_data\league_games_' + timestr + '.csv')

# print("hello rod")

