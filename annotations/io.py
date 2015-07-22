from six import string_types
from os.path import exists, isdir, join
from glob import glob
import magic
from .stims import VideoStim, AudioStim, TextStim, ImageStim
from abc import ABCMeta, abstractmethod
import pandas as pd

# Only process files with one of these extensions
EXTENSIONS = ['mp4', 'mp3', 'avi', 'jpg', 'jpeg', 'bmp', 'gif', 'txt',
              'csv', 'tsv', 'wav']


def load(source, dtype=None):
    """ Load one or more stimuli directly from file, inferring/extracting
    metadata as needed.
    Args:
        source (str or list): The location to load the stim(s) from. Can be
            the path to a directory, to a single file, or a list of filenames.
        dtype (str): The type of stim to load. If dtype is None, relies on the
            filename extension for guidance. If dtype is provided, must be
            one of 'video', 'image', 'audio', or 'text'.
    Returns: A list of Stims.
    """
    if isinstance(source, string_types):
        source = [source]

    source = [s for s in source if exists(s)]

    stims = []

    def load_file(source):
        if source.split('.')[-1] not in EXTENSIONS:
            return
        mime = magic.from_file(source, mime=True).split('/')[0]
        stim_map = {
            'image': ImageStim,
            'video': VideoStim,
            'text': TextStim,
            'audio': AudioStim
        }
        if mime in stim_map.keys():
            s = stim_map[mime](source)
            stims.append(s)

    for s in source:
        if isdir(s):
            for f in glob(join(s, '*')):
                load_file(f)
        else:
            load_file(s)

    return stims


def export(annotations, filename=None, format='fsl'):
    """ Initialize and apply an Exporter once. """
    pass


class Exporter(object):

    ''' Base exporter class. '''

    __metaclass__ = ABCMeta

    @abstractmethod
    def export(self):
        pass


class TimelineExporter(Exporter):

    ''' Exporter that handles Timelines. '''

    @staticmethod
    def timeline_to_df(timeline, format='long'):
        ''' Extracts all note data from a timeline and converts it to a
        pandas DataFrame.
        Args:
            timeline: the Timeline instance to convert
            format (str): The format of the returned DF. Either 'long'
                (default) or 'wide'. In long format, each row is a single
                key/value pair in a single Note). In wide format, each row is
                a single event, and all note values are represented in columns.
        Returns: a pandas DataFrame.
        '''
        data = []
        for onset, event in timeline.events.items():
            for note in event.notes:
                duration = event.duration or note.stim.duration
                if duration is None:
                    raise AttributeError(
                        'Duration information is missing for at least one '
                        'Event. A valid duration attribute must be set in '
                        'either the Event instance, or in the Stim '
                        'instance associated with every Note in the '
                        'Event.')
                for var, value in note.data.items():
                    data.append([onset, var, duration, value])
        data = pd.DataFrame(data,
                            columns=['onset', 'name', 'duration', 'amplitude'])
        if format == 'wide':
            data = data.pivot(index='onset', columns='name')
        return data


class FSLExporter(TimelineExporter):

    ''' Exports a Timeline as tsv files with onset, duration, and amplitude
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
                                                      'amplitude']]

        if path is not None:
            if not exists(path) or not isdir(path):
                raise IOError("The path %s does not exist or is not a "
                              "directory" % path)
            for var, d in results.items():
                filename = join(path, var + '.txt')
                d.to_csv(filename, sep='\t', index=False, header=False)
        else:
            return results
