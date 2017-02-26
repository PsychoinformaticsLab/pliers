'''
Diagnostic functions for detecting collinearity between features
'''

import numpy as np
import pandas as pd


def correlation_matrix(df):
    '''
    Returns a pandas DataFrame with the pair-wise correlations of the columns.

    Args:
        df: pandas DataFrame with columns to run diagnostics on
    '''
    columns = df.columns.tolist()
    corr = pd.DataFrame(
        np.corrcoef(df, rowvar=0), columns=columns, index=columns)
    return corr


def eigenvalues(df):
    '''
    Returns a pandas Series with eigenvalues of the correlation matrix.

    Args:
        df: pandas DataFrame with columns to run diagnostics on
    '''
    corr = np.corrcoef(df, rowvar=0)
    eigvals = np.linalg.eigvals(corr)
    return pd.Series(eigvals, df.columns, name='Eigenvalue')


def condition_indices(df):
    '''
    Returns a pandas Series with condition indices of the df columns.

    Args:
        df: pandas DataFrame with columns to run diagnostics on
    '''
    eigvals = eigenvalues(df)
    cond_idx = np.sqrt(eigvals.max() / eigvals)
    return pd.Series(cond_idx, df.columns, name='Condition index')


def variance_inflation_factors(df):
    '''
    Computes the variance inflation factor (VIF) for each column in the df.
    Returns a pandas Series of VIFs

    Args:
        df: pandas DataFrame with columns to run diagnostics on
    '''
    corr = np.corrcoef(df, rowvar=0)
    corr_inv = np.linalg.inv(corr)
    vifs = np.diagonal(corr_inv)
    return pd.Series(vifs, df.columns, name='VIF')
