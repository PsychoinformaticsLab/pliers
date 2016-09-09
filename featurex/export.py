from os.path import exists, isdir, join
from abc import ABCMeta, abstractmethod
import pandas as pd


class Exporter(object):

    ''' Base exporter class. '''

    __metaclass__ = ABCMeta

    @abstractmethod
    def export(self):
        pass


class TimelineExporter(Exporter):

    ''' Exporter that handles Timelines. '''

    @staticmethod
    def timeline_to_df(timeline, format='long', extractor=False):
        ''' Extracts all values from a timeline and converts it to a
        pandas DataFrame.
        Args:
            timeline: the Timeline instance to convert
            format (str): The format of the returned DF. Either 'long'
                (default) or 'wide'. In long format, each row is a single
                key/value pair in a single Value). In wide format, each row is
                a single event, and all Values are represented in columns.
            extractor (bool): If True, includes the name of the Extractor in
                the output (a separate column in the case of long format, and
                prepended to the column name in the case of wide).

        Returns: a pandas DataFrame.
        '''

        data = []
        for onset, event in timeline.events.items():
            for value in event.values:
                ext_name = value.extractor.name if value.extractor is not None else None
                duration = event.duration or value.stim.duration
                if duration is None:
                    raise AttributeError(
                        'Duration information is missing for at least one '
                        'Event. A valid duration attribute must be set in '
                        'either the Event instance, or in the Stim '
                        'instance associated with every Value in the '
                        'Event.')
                for var, val in value.data.items():
                    row = [onset, var, duration, val, ext_name]
                    data.append(row)

        columns=['onset', 'name', 'duration', 'value', 'extractor']
        data = pd.DataFrame(data, columns=columns)

        if format == 'wide':
            pivot_cols = ['extractor', 'name']
            data = pd.pivot_table(data, index='onset', columns=pivot_cols)
            data = data.reorder_levels([-2, -1, -3], 1)
            data = data.sortlevel(0, axis=1)
            if not extractor:
                data.columns = data.columns.droplevel()
        elif not extractor:
            data = data.drop('extractor', 1)
        return data


class FSLExporter(TimelineExporter):

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
        data = self.timeline_to_df(timeline)
        results = {}
        for var in data['name'].unique():
            results[var] = data[data['name'] == var][['onset', 'duration',
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
