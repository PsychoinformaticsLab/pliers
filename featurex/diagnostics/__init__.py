import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .collinearity import correlation_matrix
from .collinearity import eigenvalues
from .collinearity import condition_indices
from .collinearity import variance_inflation_factors
from .outliers import mahalanobis_distances
from .validity import variances

__all__ = ['collinearity', 'validity', 'outliers']

class Diagnostics(object):

    ''' Class for holding diagnostics of a design matrix '''
    def __init__(self, data, columns=None):
        self.data = data

        cols = self.data.columns if columns == None else columns
        self.eigvals = eigenvalues(self.data[cols])
        self.cond_idx = condition_indices(self.data[cols])
        self.vifs = variance_inflation_factors(self.data[cols])
        self.corr = correlation_matrix(self.data[cols])
        self.row_outliers = mahalanobis_distances(self.data[cols])
        self.column_outliers = mahalanobis_distances(self.data[cols], axis=1)
        self.variances = variances(self.data[cols])

    def show(self, stdout=True, plot=False):
        if stdout:
            print 'Collinearity summary:'
            print pd.concat([self.eigvals, self.cond_idx, self.vifs, self.corr], axis=1)

            print 'Outlier summary:'
            print self.row_outliers
            print self.column_outliers

            print 'Validity summary:'
            print self.variances
        
        if plot:
            ax = plt.axes()
            sns.heatmap(self.corr, cmap='Blues', ax=ax)
            ax.set_title('Correlation matrix')
            sns.plt.show()

            self.eigvals.plot(kind='bar', title='Eigenvalues')
            plt.show()

            self.cond_idx.plot(kind='bar', title='Condition indices')
            plt.show()

            self.vifs.plot(kind='bar', title='VIFs')
            plt.show()

            self.row_outliers.plot(kind='bar', title='Row Mahalanobis Distances')
            plt.show()

            self.column_outliers.plot(kind='bar', title='Column Mahalanobis Distances')
            plt.show()

            self.variances.plot(kind='bar', title='Variances')
            plt.show()

    def rows(self):
        pass

    def columns(self):
        pass
