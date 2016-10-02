import pandas as pd
import seaborn as sns

from .collinearity import correlation_matrix
from .collinearity import eigenvalues
from .collinearity import condition_indices
from .collinearity import variance_inflation_factors

__all__ = ['collinearity', 'validity', 'outliers']

class Diagnostics(object):

    ''' Based class for diagnostics '''
    def __init__(self, data):
        self.data = data

    def summary(self):
        '''
        Aggregates all diagnostics on the data set.
        Returns a pandas DataFrame with diagnostics.
        '''
        eigvals = eigenvalues(self.data)
        cond_idx = condition_indices(self.data)
        vifs = variance_inflation_factors(self.data)
        corr = correlation_matrix(self.data)
        
        diagnostics_df = pd.concat([eigvals, cond_idx, vifs, corr], axis=1)
        return diagnostics_df

    def show(self):
        df = self.summary()
        sns.heatmap(df, cmap='Blues')
        sns.plt.show()

    def rows(self):
        pass

    def columns(self):
        pass
