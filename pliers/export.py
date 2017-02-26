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
        ids = list(set(df.columns) & set([('stim', ''),
                                          ('onset', ''),
                                          ('duration', '')]))
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


class FSLExporter(Exporter):

    ''' Exports a Timeline as tsv files with onset, duration, and value
    columns. A separate file is created for each variable or 'condition'. '''

    def export(self, timeline, path=None):
        '''
        Args:
            timeline (Timeline): the Timeline instance to export.
            path (str): the directory to write files to.
        Returns: if path is None, returns a dictionary, where keys are variable
            names and values are pandas DataFrames. Otherwise, None.
        '''
        data = to_long_format(timeline)
        results = {}
        for var in data['stim'].unique():
            results[var] = data[data['stim'] == var][['onset', 'duration',
                                                      'value']]

        if path is not None:
            if not exists(path) or not isdir(path):
                raise IOError("The path %s does not exist or is not a "
                              "directory" % path)
            for var, d in results.items():
                filename = join(path, var + '.txt')
                d.to_csv(filename, sep='\t', index=False, header=False)
        else:
            return results
