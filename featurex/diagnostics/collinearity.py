import numpy as np
import pandas as pd

from pandas import Series


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
    vifs = variance_inflation_factors(df)
    corr = correlation_matrix(df)
    corr.columns = ['Correlation with %s' % col for col in corr.columns]
    
    diagnostics_df = pd.concat([eigvals, cond_idx, vifs, corr], axis=1)
    return diagnostics_df


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
    return Series(eigvals, df.columns, name='eigenvalue')


def condition_indices(df):
    '''
    Returns a pandas Series with condition indices of the df columns.
    
    Args:
        df: pandas DataFrame with columns to run diagnostics on
    '''
    eigvals = eigenvalues(df)
    cond_idx = np.sqrt(eigvals.max() / eigvals)
    return Series(cond_idx, name='condition index')


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
    return Series(vifs, df.columns, name='VIF')

