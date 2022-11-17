# import packages
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# from IPython.display import display

import os

# import matplotlib as mpl
# mpl.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

# Create functions to facilitate scaling, fitting, and evaluating multiple dataframes.
def evaluate_model(model, X_train, y_train, X_test, y_test, digits=4, figsize=(10, 5), params=False):
    """
    Displays evaluation metrics including classification report, confusion matrix, ROC-AUC curve.

    If the the argument 'params' is passed, will display a table of the hyperparameters used in the model.

    Args:
        df (DataFrame) : DataFrame with features to check multicollinearity on
        model (classifier object) : Type of classifier model to use.
        X_train (DataFrame) : Training data with feature variables.
        y_train (Series) : Training data with target variables.
        X_test (DataFrame) : Testing data with feature variables.
        y_test (Series) : Testing data with target variable.
        digits(int) : number of digits to display for floating point values
        figsize (int, int) : Figure dimensions. Default is (10,5)
        params(bool) : Prints table of hyperparameters used in model.

    Returns:
    """

    # Get predictions
    y_hat_test = model.predict(X_test)
    y_hat_train = model.predict(X_train)

    # Classification Report / Scores
    print("****Classification Report - Training Data****")
    print(metrics.classification_report(y_train, y_hat_train, digits=digits))

    print("****Clssification Report - Test Data****")
    print(metrics.classification_report(y_test, y_hat_test, digits=digits))

    print("****Confusion Matrix and  ROC-AUC Visualization****")
    fig, axes = plt.subplots(ncols=2, figsize=figsize)

    # Confusion Matrix
    metrics.plot_confusion_matrix(model, X_test, y_test, normalize='true', cmap='Purples', ax=axes[0])
    axes[0].set_title('Confusion Matrix')

    # Plot ROC CUrve
    metrics.plot_roc_curve(model, X_test, y_test, ax=axes[1])

    ax = axes[1]
    ax.legend()
    ax.plot([0, 1], [0, 1], ls='-')
    ax.grid()
    ax.set_title('ROC AUC Curve')

    plt.tight_layout()
    plt.show()

    if params == True:
        print("****Model Parameters****")
        params = pd.DataFrame(pd.Series(model.get_params()))
        params.columns = ['parameters']
        display(params)


def split_scale(df, target, scaler=StandardScaler()):
    """
    Creates train-test splits and scales training data.

    Args:
        df (DataFrame): DataFrame with features and target variable.
        target (str): Name of target varialbe.
        scaler(scaler object) : Scaler to use on features DataFrame, Default is StandardScaler.

    Returns:
         X_train (DataFrame): Training data with scaled features variables.
         y_train (Series): Training data with target variables
         X_test (DataFrame): Testing data with scaled feature variables.
         y_test (Series): Tesing data with target variables.
    """

    # Seperate X and y
    target = target
    y = df[target]
    X = df.drop(target, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Get list of column names
    cols = X_train.columns

    # Scale columns
    scaler = scaler
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=cols)

    return X_train, X_test, y_train, y_test


def fit_eval(model, X_train, y_train, X_test, y_test, digits=4, figsize=(10, 5), params=False):
    """
    Fits model on training data and displays classification evaluation metrics.

    Args:
        model (classifier object): Type of classification model to use.
        X_train (DataFrame): Training data with feature variables.
        y_train (Series): Training data with target variable.
        X_test (DataFrame): Testing data with feature variables.
        y_test (Series): Testing data with target variable.
        digits (int): Colormap to display correlation range; Default is 4.
        figsize (int, int): Figure dimensions. Default is (10,5)
        params (bool): Prints table of hyperparameters used in this model.

    Returns:
        model (classifier objet) : Model after fitting on training data.
    """
    model.fit(X_train, y_train)

    evaluate_model(model, X_train, y_train, X_test, y_test, digits=digits, figsize=figsize, params=params)

    return model

# Load data saved from explore_data.py
data_filt = pd.read_csv("data/logistic_regression_data", index_col=0)
data_org = pd.read_csv("data/league_games.csv", index_col=0)

# Create training and test data splits.
X_train_filt, X_test_filt, y_train_filt, y_test_filt = split_scale(data_filt, 'blueWins')
X_train_org, X_test_org, y_train_org, y_test_org = split_scale(data_org, 'blueWins')


# Fit and evaluate
log_reg_model = fit_eval(LogisticRegressionCV(random_state=42), X_train_filt, y_train_filt, X_test_filt, y_test_filt)


# Create parameter grid for Logistic Regression gridsearch and fit to data
log_reg = LogisticRegression(random_state=42)

params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1e6, 1e12],
          'penalty': ['l1', 'l2', 'elastic_net'],
          'fit_intercept': [True, False],
          'solver': ["liblinear", "newton-cg", "lbfgs", "sag", "saga"],
          'class_weight': ['balanced']
          }
log_grid =GridSearchCV(log_reg, params, scoring='recall_macro')
log_grid.fit(X_train_filt, y_train_filt)

#Print the best params for log_grid
log_grid.best_params_

#Evaluate best estimating model
evaluate_model(log_grid.best_estimator_, X_train_filt, y_train_filt, X_test_filt, y_test_filt, params=True)


# See a decrease in recall score, let's tune hyperparameters:
# Create parameter grid for Logistic Regression gridsearch and fit to data (second time)
log_reg_ref = LogisticRegression(random_state=42)

params = {'C': [0.0001, 0.001],
          'penalty': ['l1', 'l2', 'elastic_net'],
          'solver':["liblinear", "newton-cg", "lbfgs", "sag", "saga"],
          'class_weight': ['balanced']}
log_grid_refined = GridSearchCV(log_reg_ref, params, scoring='recall_macro')
log_grid_refined.fit(X_train_filt, y_train_filt)

# Print best estimating model (second time)
evaluate_model(log_grid_refined.best_estimator_, X_train_filt, y_train_filt, X_test_filt, y_test_filt, params=True)

#this doesnt fix the improve recall score, log_reg is still the best model
