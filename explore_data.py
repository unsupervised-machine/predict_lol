#import packages
import pandas as pd
import numpy as np
np.random.seed(42)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Load data and display basic info
df = pd.read_csv("data/league_games.csv")

df.info()
df.isnull().values.any()
pd.set_option('display.max_columns', 40)
df.head()