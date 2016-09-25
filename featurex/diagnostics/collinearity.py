import numpy as np
import pandas as pd

from pandas import Series
from statsmodels.stats.outliers_influence import variance_inflation_factor

def collinearity_diagnostics_matrix(df):
    '''
    Aggregates diagnostics related to collinearity.
    Returns a pandas DataFrame with collinearity diagnostics, modeled 
    after SPSS regression diagnostics.

    Args:
        df: pandas DataFrame with columns to run diagnostics on
    '''
    eigvals = eigenvalues(df)
    cond_idx = condition_indices(df)
    pass


def correlation_matrix(df):
    '''
    Returns a pandas DataFrame with the pair-wise correlations of the columns.

    Args:
        df: pandas DataFrame with columns to run diagnostics on
    '''
    columns = df.columns.tolist()
    corr_df = pd.DataFrame(np.corrcoef(df, rowvar=0), columns=columns, index=columns)
    return corr_df


def eigenvalues(df):
    '''
    Returns a pandas Series with eigenvalues of the correlation matrix.
    
    Args:
        df: pandas DataFrame with columns to run diagnostics on
    '''
    corr = np.corrcoef(df, rowvar=0)
    eigvals = np.linalg.eigvals(corr)
    return Series(eigvals, df.columns)


def condition_indices(df):
    '''
    Returns a pandas Series with condition indices of the df columns.
    
    Args:
        df: pandas DataFrame with columns to run diagnostics on
    '''
    eigvals = eigenvalues(df)
    cond_idx = np.sqrt(eigvals.max() / eigvals)
    return cond_idx


def vifs(df):
    '''
    Computes the variance inflation factor (VIF) for each column in the df.
    Returns a numpy array of VIFs

    Args:
        df: pandas DataFrame with columns to run diagnostics on
    '''
    # Calculation 1
    vifs = []
    for index in range(len(df.columns)):
        vifs.append(variance_inflation_factor(df.values, index))

    # Calculation 2 (doesn't use statsmodels)
    corr_df = correlation_matrix(df)
    corr_inv = np.linalg.inv(corr_df.values)
    vifs = np.diagonal(corr_inv)

    # Need to figure out which calculation is correct
    return Series(vifs, df.columns)

