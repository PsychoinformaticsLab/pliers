from abc import ABCMeta, abstractmethod, abstractproperty
from six import with_metaclass, string_types
from pliers.utils import listify
from pliers.stimuli import _get_stim_class
from pliers.stimuli.audio import AudioStim
from pliers.stimuli.text import ComplexTextStim


class CompoundStim(object):

    ''' A container for an arbitrary set of Stims.
    Args:
        stims (Stim or list): a single Stim (of any type) or a list of Stims.

    '''
    _allowed_types = None
    _allow_multiple = True

    def __init__(self, stims):

        self.stims = []
        _type_dict = {}
        for s in stims:
            stim_cl, self_cl = s.__class__.__name__, self.__class__.__name__
            if self._allowed_types and not isinstance(s, self._allowed_types):
                raise ValueError("A stim of class %s was passed, but the %s "
                                 "class does not support a component of this "
                                 "type." % (stim_cl, self_cl))
            if self._allow_multiple or stim_cl not in _type_dict:
                _type_dict[stim_cl] = 1
                self.stims.append(s)
            else:
                msg = "Multiple components of same type not allowed, but "+ \
                      "a stim of type %s already exists in this %s." % (stim_cl, self_cl)
                raise ValueError(msg)
        self.name = '&'.join([s.name for s in self.stims])

    def get_stim(self, type_, return_all=False):
        ''' Returns component Stims of the specified type.
        Args:
            type_ (str or Stim class): the desired Stim subclass to return.
            return_all (bool): when True, returns all stims that matched the
                specified type as a list. When False (default), returns only
                the first matching Stim.
        Returns:
            If return_all is True, a list of matching Stims (or an empty list
            if no Stims match). If return_all is False, returns the first
            matching Stim, or None if no Stims match.
        '''
        if isinstance(type_, string_types):
            type_ = _get_stim_class(type_)
        matches = []
        for s in self.stims:
            if isinstance(s, type_):
                if not return_all:
                    return s
                matches.append(s)
        if not matches:
            return [] if return_all else None
        return matches

    def __getattr__(self, attr):
        try:
            stim = _get_stim_class(attr)
        except:
            raise AttributeError()
        return self.get_stim(stim)


class TranscribedAudioCompoundStim(CompoundStim):

    ''' An AudioStim with an associated text transcription.
    Args:
        filename (str): The path to the audio clip.
        transcription (str or ComplexTextStim): the associated transcription.
            If a string, this is interpreted as the name of a file containing
            data needed to initialize a new ComplexTextStim. Otherwise, must
            pass an existing ComplexTextStim instance.
        kwargs (dict): optional keywords passed to the ComplexTextStim
            initializer if transcription argument is a string.
    '''
    _allowed_types = (AudioStim, ComplexTextStim)
    _allow_multiple = False

    # def __init__(self, filename, transcription, onset=None, **kwargs):
    def __init__(self, audio, text):
        super(TranscribedAudioCompoundStim, self).__init__(stims=[audio, text])

