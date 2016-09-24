import numpy as np
import pandas as pd

from pandas import Series
from statsmodels.stats.outliers_influence import variance_inflation_factor

def collinearity_diagnostics_matrix(df):
    '''
    Returns a pandas DataFrame with collinearity diagnostics, modeled after SPSS

    Args:
        df: pandas DataFrame with columns to run diagnostics on
    '''
    pass


def correlation_matrix(df):
    '''
    Returns correlation matrix, as a pandas DataFrame, for the columns

    Args:
        df: pandas DataFrame with columns to run diagnostics on
    '''
    columns = df.columns.tolist()
    cov_df = pd.DataFrame(np.corrcoef(df, rowvar=0), columns=columns, index=columns)
    return cov_df


def variances(df):
    '''
    Computes variance of columns, as a pandas Series for each column

    Args:
        df: pandas DataFrame with columns to run diagnostics on
    '''
    return df.var(axis=0)


def eigenvalues(df):
    '''
    Returns a pandas DataFrame with collinearity diagnostics
    
    Args:
        df: pandas DataFrame with columns to run diagnostics on
    '''
    #return np.linalg.eigvals(df)
    pass


def condition_indices(df):
    '''
    Returns a pandas DataFrame with collinearity diagnostics
    
    Args:
        df: pandas DataFrame with columns to run diagnostics on
    '''
    # for i, name in enumerate(X):
    #     if name == "const":
    #         continue
    #     norm_x[:,i] = X[name]/np.linalg.norm(X[name])
    # norm_xtx = np.dot(norm_x.T,norm_x)


    # # Then, we take the square root of the ratio of the biggest to the smallest eigen values. 

    # eigs = np.linalg.eigvals(norm_xtx)
    # condition_number = np.sqrt(eigs.max() / eigs.min())
    # print(condition_number)
    pass


def vifs(df):
    '''
    Computes the variance inflation factor (VIF) for each column in the df
    Returns a numpy array of VIFs

    Args:
        df: pandas DataFrame with columns to run diagnostics on
    '''
    vifs = []
    for index, name in enumerate(df.columns):
        vifs.append(variance_inflation_factor(df.values, index))
    return Series(vifs, df.columns)

