#import packages
import pandas as pd
import numpy as np
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



np.random.seed(42)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Load data and display basic info
df = pd.read_csv("data/league_games.csv", index_col=0)

df.info()
df.isnull().values.any()
pd.set_option('display.max_columns', 40)
df.head()


df_org = df.copy() #original dataframe preserved
#grab all column names
col_names = pd.Series(df.columns)

#target variable = bluewins and non engineer variables
df_filt = df.filter(items=['blueWins', 'blueWardsPlaced', 'blueWardsDestroyed',
                            'blueFirstBlood', 'blueKills', 'blueDeaths',
                            'blueAssists', 'blueDragons', 'blueHeralds',
                           'blueTowersDestroyed', 'blueTotalMinionKilled', 'blueTotalJungleMinionsKilled',
                           'redWardsPlaced', 'redWardsDestroyed', 'redFirstBlood',
                           'redKills', 'redDeaths', 'redAssists',
                           'redTotalMinionsKilled', 'redTotalJungleMinionsKilled'
                           ])

df_filt.head()

#check for any multicollinearity; anything with score over 10 is too high
vif_data = pd.DataFrame()
vif_data["feature"] = df_filt.columns
vif_data["VIF"] = [variance_inflation_factor(df_filt, i) for i in range(df_filt.shape[1])]



#create function to visualize correlation
def corr_heatmap(df, digits=3, cmap='coolwarm'):
    """
    Create a correclation heatmap to easily visualize multicollinearity.
    Args:
        df(DataFrame) : DataFrame with features to check multicollinearity on.
        digits (int) : Number of decimal places to display.
        cmap (str) : Colormap to display correlation range.

    Returns:
        fig: Matplotlib Figure
        ax Matplotlib Axis
    """
    # Create correlation matrix from dataframe
    correl = df.corr().round(digits)
    correl

    #Replace the top triangle of matrix with 0's
    mask = np.zeros_like(correl)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=((len(df.columns)), (len(df.columns))))
    sns.heatmap(correl, annot=True, ax=ax, cmap=cmap, vmin=-1, vmax=1, mask = mask)

    return fig, ax


def visual_eda(df, target, col):
    """
    Plots a histogram + Kenel Density Estimate (KDE), boxplot, scatterplot with linear regression.
    To visualize the shape of the datam outliersm and check column's correlation with target variable.

    Args:
        df (DataFrame) : DataFrame containing the col to plot
        target (str) : Name of the target variable.
        col (str) : Name of the column to plot

    Returns:
        fig : Matplotlib Figure
        gs: Matplotlib Gridspec
    """
    data = df[col].copy()
    name = col

    # Calc mean and median
    median = data.median().round(2)
    mean = data.mean().round(2)

    #Create gridspec for plots
    fig = plt.figure(figsize=(11, 6))
    gs = GridSpec(nrows = 2, ncols = 2)

    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[1,0])
    ax2 = fig.add_subplot(gs[:,1])

    #Plot distribution
    sns.histplot(data, alpha=0.5, stat='density', ax=ax0)
    sns.kdeplot(data, color = 'green', label = 'KDE', ax=ax0)
    ax0.set(ylabel='Density', title=name)
    ax0.set_title(F"Distribution of {name}")
    ax0.axvline(median, label=f'median= {median:,}', color='black')
    ax0.axvline(mean,label=f'mean={mean:,}',color='black',ls=':')
    ax0.legend()

    #Plot Boxplot
    sns.boxplot(x= data, ax=ax1)
    ax1.set_title(F"Box Plot of {name}")

    #Plot Scatterplot to illustrate linearity
    sns.regplot(data=df, x=col, y=target, line_kws={"color": "red"}, ax=ax2)
    ax2.set_title(F"Scatter Plot of {name}")

    #Tweak layout and display
    fig.tight_layout()
    # fig.savefig("sample.png")
    return fig, gs



# Create correlation heatmap for df_org
corr_heatmap(df_org)
# Create correlation heatmap for df_filt.
corr_heatmap(df_filt)


#remove variables with multicollinearity
df_filt.drop(columns=['redKills', 'redDeaths', 'redFirstBlood'], inplace=True)
df_filt.to_csv("data/logistic_regression_data")

#Check for target variable imbalance
fig, ax = plt.subplots(figsize=(8,6))
sns.countplot(x='blueWins', data=df_filt, hue='blueWins', palette='cool_r')


#See how xp, gold are correlated with blue wins (both teams)
visual_eda(df_org, 'blueWins', 'blueTotalExperience')
visual_eda(df_org, 'blueWins', 'redTotalExperience')
visual_eda(df_org, 'blueWins', 'blueTotalGold')
visual_eda(df_org, 'blueWins', 'redTotalGold')


