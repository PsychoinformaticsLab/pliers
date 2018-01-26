import pandas as pd
import numpy as np
from pliers.utils import attempt_to_import, verify_dependencies
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from numpy.linalg import LinAlgError


sns = attempt_to_import('seaborn')


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


def mahalanobis_distances(df, axis=0):
    '''
    Returns a pandas Series with Mahalanobis distances for each sample on the
    axis.

    Note: does not work well when # of observations < # of dimensions
    Will either return NaN in answer
    or (in the extreme case) fail with a Singular Matrix LinAlgError

    Args:
        df: pandas DataFrame with columns to run diagnostics on
        axis: 0 to find outlier rows, 1 to find outlier columns
    '''
    df = df.transpose() if axis == 1 else df
    means = df.mean()
    try:
        inv_cov = np.linalg.inv(df.cov())
    except LinAlgError:
        return pd.Series([np.NAN] * len(df.index), df.index,
                         name='Mahalanobis')
    dists = []
    for i, sample in df.iterrows():
        dists.append(mahalanobis(sample, means, inv_cov))

    return pd.Series(dists, df.index, name='Mahalanobis')


def variances(df):
    '''
    Returns a pandas Series with variances for each column

    Args:
        df: pandas DataFrame with columns to run diagnostics on
    '''
    return pd.Series(df.var(axis=0), name='Variances')


class Diagnostics(object):
    defaults = {
        'Eigenvalues': (lambda x: x < 0.05),
        'ConditionIndices': (lambda x: x > 20),
        'VIFs': (lambda x: x > 10),
        'CorrelationMatrix': (lambda x: x > 0.5),
        'RowMahalanobisDistances': (lambda x: x > 5),
        'ColumnMahalanobisDistances': (lambda x: x > 5),
        'Variances': (lambda x: x < 0.15)
    }

    ''' Class for holding diagnostics of a design matrix '''

    def __init__(self, data, columns=None):
        self.data = data

        cols = self.data.columns if columns is None else columns
        self.results = {}
        self.results['Eigenvalues'] = eigenvalues(self.data[cols])
        self.results['ConditionIndices'] = condition_indices(self.data[cols])
        self.results['VIFs'] = variance_inflation_factors(self.data[cols])
        self.results['CorrelationMatrix'] = correlation_matrix(self.data[cols])
        self.results['RowMahalanobisDistances'] = mahalanobis_distances(
            self.data[cols])
        self.results['ColumnMahalanobisDistances'] = mahalanobis_distances(
            self.data[cols], axis=1)
        self.results['Variances'] = variances(self.data[cols])

    def summary(self, stdout=True, plot=False):
        '''
        Displays diagnostics to the user

        Args:
            stdout (bool): print results to the console
            plot (bool): use Seaborn to plot results
        '''
        if stdout:
            print('Collinearity summary:')
            print(pd.concat([self.results['Eigenvalues'],
                             self.results['ConditionIndices'],
                             self.results['VIFs'],
                             self.results['CorrelationMatrix']],
                            axis=1))

            print('Outlier summary:')
            print(self.results['RowMahalanobisDistances'])
            print(self.results['ColumnMahalanobisDistances'])

            print('Validity summary:')
            print(self.results['Variances'])

        if plot:
            verify_dependencies('seaborn')
            for key, result in self.results.items():
                if key == 'CorrelationMatrix':
                    ax = plt.axes()
                    sns.heatmap(result, cmap='Blues', ax=ax)
                    ax.set_title(key)
                    sns.plt.show()
                else:
                    result.plot(kind='bar', title=key)
                    plt.show()

    def flag(self, diagnostic, thresh=None):
        '''
        Returns indices of diagnostic that satisfy (return True from) the
        threshold predicate. Will use class-level default threshold if
        None provided.

        Args:
            diagnostic (str): name of the diagnostic
            thresh (func): threshold function (boolean predicate) to apply to
            each element
        '''
        if thresh is None:
            thresh = self.defaults[diagnostic]

        result = self.results[diagnostic]
        if isinstance(result, pd.DataFrame):
            if diagnostic == 'CorrelationMatrix':
                result = result.copy()
                np.fill_diagonal(result.values, 0)
            return result.applymap(thresh).sum().nonzero()[0]
        else:
            return result.apply(thresh).nonzero()[0]

    def flag_all(self, thresh_dict=None, include=None, exclude=None):
        '''
        Returns indices of (rows, columns) that satisfy flag() on any
        diagnostic. Uses user-provided thresholds in thresh_dict/

        Args:
            thresh_dict (dict): dictionary of diagnostic->threshold functions
            include (list): optional sublist of diagnostics to flag
            exclude (list): optional sublist of diagnostics to not flag
        '''
        if thresh_dict is None:
            thresh_dict = {}
        row_idx = set()
        col_idx = set()
        include = self.results if include is None else include
        include = list(
            set(include) - set(exclude)) if exclude is not None else include
        for diagnostic in include:
            if diagnostic in thresh_dict:
                flagged = self.flag(diagnostic, thresh_dict[diagnostic])
            else:
                flagged = self.flag(diagnostic)

            if diagnostic == 'RowMahalanobisDistances':
                row_idx = row_idx.union(flagged)
            else:
                col_idx = col_idx.union(flagged)

        return sorted(list(row_idx)), sorted(list(col_idx))
