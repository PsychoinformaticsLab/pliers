''' Filters that operate on TextStim inputs. '''

from copy import deepcopy

from pliers.stimuli import AudioStim
from pliers.utils import attempt_to_import, verify_dependencies
from .base import Filter, TemporalTrimmingFilter

librosa = attempt_to_import('librosa')

class AudioFilter(Filter):

    ''' Base class for all audio filters. '''

    _input_type = AudioStim


class AudioTrimmingFilter(TemporalTrimmingFilter, AudioFilter):
    pass


class AudioResamplingFilter(AudioFilter):
    
    ''' Librosa-based audio resampling Filter.
        Uses librosa.core.resample function.

    Args:
        target_sr (float): Target sampling rate (in Hz).
        resample_type (str): Type of resampling. Must be one of 
            'kaiser_best', 'kaiser_fast', 'scipy', 'fft' or 
            'polyphase'. See librosa.core.resample documentation for 
            more details.
        librosa_kwargs: Optional keyword args passed onto the
            librosa resampling function.
    '''
    
    _log_attributes = ('target_sr', 'resample_type')
    
    def __init__(self, target_sr=44100, resample_type='kaiser_best',
                 **librosa_kwargs):
        verify_dependencies(['librosa'])
        self.target_sr = target_sr
        self.resample_type = resample_type
        self.librosa_kwargs = librosa_kwargs
        super().__init__()

    def _filter(self, stim):
        resampled_stim = deepcopy(stim)
        resampled_stim.data = librosa.core.resample(y=stim.data,
                                          orig_sr=stim.sampling_rate,
                                          target_sr=self.target_sr,
                                          resample_type=self.resample_type,
                                          **self.librosa_kwargs)
        resampled_stim.sampling_rate = self.target_sr
        return resampled_stim