''' Base Diagnostics class. '''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .collinearity import correlation_matrix
from .collinearity import eigenvalues
from .collinearity import condition_indices
from .collinearity import variance_inflation_factors
from .outliers import mahalanobis_distances
from .validity import variances

__all__ = [
    'correlation_matrix',
    'eigenvalues',
    'condition_indices',
    'variance_inflation_factors',
    'mahalanobis_distances',
    'variances'
]


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
