''' Base class for all Stims and associated functionality. '''


from abc import ABCMeta, abstractmethod
from os.path import isdir, join, basename, realpath, isfile
from glob import glob
from six import with_metaclass, string_types
from six.moves.urllib.request import urlopen
from six.moves.urllib.parse import urlparse
from collections import namedtuple
from contextlib import contextmanager
from pliers import config
from pliers.utils import isiterable
import pandas as pd
import os
import tempfile
import base64


class Stim(with_metaclass(ABCMeta)):

    ''' Base class for all classes in the Stim hierarchy.

    Args:
        filename (str): Path to input file, if one exists.
        onset (float): Optional onset of the Stim (in seconds) with
            respect to some more general context or timeline the user wishes
            to keep track of.
        duration (float): Optional duration of the Stim, in seconds.
        order (int): Optional order of Stim within some broader context.
        name (str): Optional name to give the Stim instance. If None is
            provided, the name will be derived from the filename if one is
            defined. If no filename is defined, name will be an empty string.
    '''

    def __init__(self, filename=None, onset=None, duration=None, order=None,
                 name=None, url=None):

        self.filename = filename
        self.onset = onset
        self.duration = duration
        self.order = order
        self._history = None
        self.url = url

        if name is None:
            name = '' if self.filename is None else basename(self.filename)
        self.name = name

    @abstractmethod
    def save(self, path):
        pass

    @contextmanager
    def get_filename(self):
        ''' Return the source filename of the current Stim. '''
        if self.filename is None or not os.path.exists(self.filename):
            tf = tempfile.mktemp() + self._default_file_extension
            self.save(tf)
            yield tf
            os.remove(tf)
        else:
            yield self.filename

    @property
    def history(self):
        ''' Return stimulus history. '''
        return self._history

    @history.setter
    def history(self, history):
        self._history = history

    def __hash__(self):
        return hash((self.filename, self.name, self.onset, self.duration,
                     self.order, self.history))


def _get_stim_class(name):
    # Takes a lowercase variable name (e.g., 'video' or 'complex_text') and
    # attempts to map it to a valid pliers Stim class (e.g., VideoStim or
    # ComplexTextStim).
    name = name.lower().replace('_', '')

    if not name.endswith('stim'):
        name += 'stim'

    import pliers
    stims = pliers.stimuli.__all__

    for a in stims:
        if a.lower() == name.lower():
            return getattr(pliers.stimuli, a)

    raise KeyError("No Stim class matches '%s' (case-insensitive)." % name)


def load_stims(source, dtype=None, fail_silently=False):
    """ Load one or more stimuli directly from file, inferring/extracting
    metadata as needed.

    Args:
        source (str or list): The location to load the stim(s) from. Can be
            the path to a directory, to a single file, or a list of filenames.
        dtype (str): The type of stim to load. If dtype is None, relies on the
            filename extension for guidance. If dtype is provided, must be
            one of 'video', 'image', 'audio', or 'text'.
        fail_silently (bool): If True do not raise error when trying to load a
            missing stimulus from a list of sources.

    Returns: A list of Stims.
    """
    from .video import VideoStim, ImageStim
    from .audio import AudioStim
    from .text import TextStim

    if isinstance(source, string_types):
        return_list = False
        source = [source]
    else:
        return_list = True

    stims = []

    stim_map = {
        'image': ImageStim,
        'video': VideoStim,
        'text': TextStim,
        'audio': AudioStim
    }

    def load_file(source):
        source = realpath(source)
        import magic  # requires libmagic, so import here
        mime = magic.from_file(source, mime=True)
        if not isinstance(mime, string_types):
            mime = mime.decode('utf-8')
        mime = mime.split('/')[0]
        if mime in stim_map.keys():
            s = stim_map[mime](source)
            stims.append(s)

    def load_url(source):
        try:
            main_type = urlopen(source).info().get_content_maintype()  # Py3
        except:
            main_type = urlopen(source).info().getmaintype()  # Py2
        if main_type in stim_map.keys():
            s = stim_map[main_type](url=source)
            stims.append(s)

    for s in source:
        if bool(urlparse(s).scheme):
            load_url(s)
        elif isdir(s):
            for f in glob(join(s, '*')):
                if isfile(f):
                    load_file(f)
        elif isfile(s):
            load_file(s)
        else:
            if not (return_list and fail_silently):
                raise IOError("File not found")

    if return_list:
        return stims

    return stims[0]


def _get_bytestring(stim, encoding='utf-8'):
    if stim._bytestring is None:
        with stim.get_filename() as filename:
            with open(filename, 'rb') as f:
                data = f.read()
                stim._bytestring = base64.b64encode(data).decode(encoding=encoding)

    return stim._bytestring


def _log_transformation(source, result, trans=None, implicit=False):

    if result is None or not config.get_option('log_transformations') or \
            (trans is not None and not trans._loggable):
        return result

    if isiterable(result):
        return (_log_transformation(source, r, trans) for r in result)

    values = [source.name, source.filename, source.__class__.__name__]
    if isinstance(result, Stim):
        values.extend([result.name, result.filename])
    else:
        values.extend(['', ''])
    values.append(result.__class__.__name__)
    if trans is not None:
        values.append(trans.__class__.__name__)
        tr_attrs = [getattr(trans, attr) for attr in trans._log_attributes]
        values.append(str(dict(zip(trans._log_attributes, tr_attrs))))
    else:
        values.append(['', ''])
    parent = source.history
    string = str(parent) if parent else values[2]
    string += '->%s/%s' % (values[6], values[5])
    values.extend([string, parent])
    values.append(implicit)
    result.history = TransformationLog(*values)
    return result


_trans_log = namedtuple('TransformationLog', "source_name source_file " +
                        "source_class result_name result_file result_class " +
                        " transformer_class transformer_params string " +
                        "parent implicit")


class TransformationLog(_trans_log):

    '''A namedtuple that stores information about a single transformation. '''

    __slots__ = ()

    def __str__(self):
        return self.string

    def to_df(self):
        def _append_row(rows, history):
            rows.append(history[:-3])
            if history.parent:
                _append_row(rows, history.parent)
            return rows
        rows = _append_row([], self)[::-1]
        return pd.DataFrame(rows, columns=self._fields[:-3])
