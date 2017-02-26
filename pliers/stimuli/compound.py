''' A CompoundStim class represents a combination of constituent Stim classes.
'''

from six import string_types
from pliers.utils import listify
from .base import _get_stim_class
from .audio import AudioStim
from .text import ComplexTextStim


class CompoundStim(object):

    ''' A container for an arbitrary set of Stim elements.
    Args:
        elements (Stim or list): a single Stim (of any type) or a list of
        elements.
    '''

    _allowed_types = None
    _allow_multiple = True
    _primary = None

    def __init__(self, elements):

        self.elements = []
        self.history = None
        _type_dict = {}
        for s in elements:
            stim_cl, self_cl = s.__class__.__name__, self.__class__.__name__
            if self._allowed_types and not isinstance(s, self._allowed_types):
                raise ValueError("A stim of class %s was passed, but the %s "
                                 "class does not allow elements of this "
                                 "type." % (stim_cl, self_cl))
            if self._allow_multiple or stim_cl not in _type_dict:
                _type_dict[stim_cl] = 1
                self.elements.append(s)
            else:
                msg = "Multiple components of same type not allowed, and " + \
                      "a stim of type %s already exists in this %s." % (stim_cl, self_cl)
                raise ValueError(msg)

        if self._primary is not None:
            primary = self.get_stim(self._primary)
            self.name = primary.name
            self.filename = primary.filename

        else:
            self.name = '&'.join([s.name for s in self.elements])[:255]
            self.filename = None

    def get_stim(self, type_, return_all=False):
        ''' Returns component elements of the specified type.
        Args:
            type_ (str or Stim class): the desired Stim subclass to return.
            return_all (bool): when True, returns all elements that matched the
                specified type as a list. When False (default), returns only
                the first matching Stim.
        Returns:
            If return_all is True, a list of matching elements (or an empty
            list if no elements match). If return_all is False, returns the
            first matching Stim, or None if no elements match.
        '''
        if isinstance(type_, string_types):
            type_ = _get_stim_class(type_)
        matches = []
        for s in self.elements:
            if isinstance(s, type_):
                if not return_all:
                    return s
                matches.append(s)
        if not matches:
            return [] if return_all else None
        return matches

    def get_types(self):
        ''' Return tuple of types of all available Stims. '''
        return tuple(set([e.__class__ for e in self.elements]))

    def has_types(self, types, all_=True):
        ''' Check whether the current component list matches all Stim types
        in the types argument.
        Args:
            types (Stim, list): a Stim class or iterable of Stim classes.
            all_ (bool): if True, all input types must match; if False, at
                least one input type must match.
        Return:
            True if all passed types match at least one Stim in the component
            list, otherwise False.
        '''
        func = all if all_ else any
        return func([self.get_stim(t) for t in listify(types)])

    def __getattr__(self, attr):
        try:
            stim = _get_stim_class(attr)
        except:
            return self.__getattribute__(attr)
        return self.get_stim(stim)


class TranscribedAudioCompoundStim(CompoundStim):

    ''' An AudioStim with an associated text transcription.
    Args:
        filename (str): The path to the audio clip.
        audio (AudioStim): An AudioStim containing the audio content.
        text (ComplexTextStim): A ComplexTextStim containing the transcribed
            text (and associated timing information).
    '''
    _allowed_types = (AudioStim, ComplexTextStim)
    _allow_multiple = False
    _primary = AudioStim

    # def __init__(self, filename, transcription, onset=None, **kwargs):
    def __init__(self, audio, text):
        super(TranscribedAudioCompoundStim, self).__init__(elements=[audio, text])
