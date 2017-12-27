from os.path import exists, isdir, join
from abc import ABCMeta, abstractmethod
import pandas as pd
from six import with_metaclass
from pliers.extractors.base import ExtractorResult


class Exporter(with_metaclass(ABCMeta)):

    ''' Base exporter class. '''

    @abstractmethod
    def export(self):
        pass


def to_long_format(df):
    ''' Convert from wide to long format, making each row a single
    feature/value pair.

    Args:
        df (DataFrame): a timeline that is currently in wide format
    '''
    if isinstance(df, ExtractorResult):
        df = df.to_df()

    if isinstance(df.columns, pd.core.index.MultiIndex):
        ids = list(filter(lambda x: x[1] is '', df.columns))
        variables = ['extractor', 'feature']
    else:
        df = df.reset_index() if not isinstance(
            df.index, pd.Int64Index) else df
        ids = list(set(df.columns) & set(['stim', 'onset', 'duration']))
        variables = 'feature'

    values = list(set(df.columns) - set(ids))
    converted = pd.melt(df, id_vars=ids, value_vars=values, var_name=variables)
    converted.columns = [
        c[0] if isinstance(c, tuple) else c for c in converted.columns]
    return converted
