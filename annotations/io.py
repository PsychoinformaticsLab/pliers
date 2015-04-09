from six import string_types
from os.path import exists, isdir, join
from glob import glob
import magic
from .stimuli import VideoStim, AudioStim, TextStim, ImageStim

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


class ExportFormat(object):
    pass


class DataSource(object):
    pass
    """ Data sources required to produce an annotation, e.g., external
    corpora. """
