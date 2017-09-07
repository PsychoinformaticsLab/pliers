''' Extractors that operate on AudioStim inputs. '''

from pliers.stimuli.audio import AudioStim
from pliers.stimuli.text import ComplexTextStim
from pliers.extractors.base import Extractor, ExtractorResult
import numpy as np
from scipy import fft

try:
    import librosa
except ImportError:
    librosa = None


class AudioExtractor(Extractor):

    ''' Base Audio Extractor class; all subclasses can only be applied to
    audio. '''
    _input_type = AudioStim


class STFTAudioExtractor(AudioExtractor):

    ''' Short-time Fourier Transform extractor.
    Args:
        frame_size (float): The width of the frame/window to apply an FFT to,
            in seconds.
        hop_size (float): The step size to increment the window by on each
            iteration, in seconds (effectively, the sampling rate).
        freq_bins (list or int): The set of bins or frequency bands to extract
            power for. If an int is passed, this is the number of bins
            returned, with each bin spanning an equal range of frequencies.
            E.g., if bins=5 and the frequency spectrum runs from 0 to 20KHz,
            each bin will span 4KHz. If a list is passed, each element must be
            a tuple or list of lower and upper frequency bounds. E.g., passing
            [(0, 300), (300, 3000)] would compute power in two bands, one
            between 0 and 300Hz, and one between 300Hz and 3KHz.
        spectrogram (bool): If True, plots a spectrogram of the results.

    Notes: code adapted from
    http://stackoverflow.com/questions/2459295/invertible-stft-and-istft-in-python
    '''

    _log_attributes = ('frame_size', 'hop_size', 'freq_bins')

    def __init__(self, frame_size=0.5, hop_size=0.1, freq_bins=5,
                 spectrogram=False):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.spectrogram = spectrogram
        self.freq_bins = freq_bins
        super(STFTAudioExtractor, self).__init__()

    def _stft(self, stim):
        x = stim.data
        framesamp = int(self.frame_size*stim.sampling_rate)
        hopsamp = int(self.hop_size*stim.sampling_rate)
        w = np.hanning(framesamp)
        X = np.array([fft(w*x[i:(i+framesamp)])
                      for i in range(0, len(x)-framesamp, hopsamp)])
        nyquist_lim = int(X.shape[1]//2)
        X = np.log(X[:, :nyquist_lim])
        X = np.absolute(X)
        if self.spectrogram:
            import matplotlib.pyplot as plt
            bins = np.fft.fftfreq(framesamp, d=1./stim.sampling_rate)
            bins = bins[:nyquist_lim]
            plt.imshow(X.T, origin='lower', aspect='auto',
                       interpolation='nearest', cmap='RdYlBu_r',
                       extent=[0, stim.duration, bins.min(), bins.max()])
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.colorbar()
            plt.show()
        return X

    def _extract(self, stim):
        data = self._stft(stim)
        time_bins = np.arange(0., stim.duration-self.frame_size, self.hop_size)

        if isinstance(self.freq_bins, int):
            bins = []
            bin_size = data.shape[1] / self.freq_bins
            for i in range(self.freq_bins):
                bins.append((i*bin_size, (i+1)*bin_size))
            self.freq_bins = bins

        features = ['%d_%d' % fb for fb in self.freq_bins]
        offset = 0.0 if stim.onset is None else stim.onset
        index = [tb + offset for tb in time_bins]
        values = np.zeros((len(index), len(features)))
        for i, fb in enumerate(self.freq_bins):
            start, stop = fb
            values[:, i] = data[:, start:stop].mean(1)
        values[np.isnan(values)] = 0.
        values[np.isinf(values)] = 0.
        return ExtractorResult(values, stim, self, features=features,
                               onsets=index, durations=self.hop_size)


class MeanAmplitudeExtractor(Extractor):
    ''' Mean amplitude extractor for blocks of audio with transcription. '''

    _input_type = (AudioStim, ComplexTextStim)

    def _extract(self, stim):

        amps = stim.audio.data
        sampling_rate = stim.audio.sampling_rate
        elements = stim.complex_text.elements
        values, onsets, durations = [], [], []

        for i, el in enumerate(elements):
            onset = sampling_rate * el.onset
            onsets.append(onset)
            duration = sampling_rate * el.duration
            durations.append(duration)

            r_onset = np.round(onset).astype(int)
            r_offset = np.round(onset+duration).astype(int)
            if not r_offset <= amps.shape[0]:
                raise Exception('Block ends after data.')

            mean_amplitude = np.mean(amps[r_onset:r_offset])
            values.append(mean_amplitude)

        return ExtractorResult(values, stim, self, features=['mean_amplitude'],
                               onsets=onsets, durations=durations)


class LibrosaFeatureExtractor(AudioExtractor):

    ''' A generic class for all audio extractors using the librosa library. '''

    def __init__(self):
        if librosa is None:
            raise ImportError("librosa is required to create a "
                              "LibrosaFeatureExtractor, but could not be "
                              "successfully imported. Please make sure it is "
                              "installed.")
        super(LibrosaFeatureExtractor, self).__init__()


class SpectralCentroidExtractor(LibrosaFeatureExtractor):

    ''' '''

    def __init__(self):
        super(SpectralCentroidExtractor, self).__init__()

    def _extract(self, stim):
        pass


class RMSEExtractor(LibrosaFeatureExtractor):

    ''' Extracts root mean square (RMS) energy from audio. '''

    _log_attributes = ('frame_length', 'hop_length', 'center', 'pad_mode')

    def __init__(self, frame_length=2048, hop_length=512, center=True,
                 pad_mode='reflect'):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        super(RMSEExtractor, self).__init__()

    def _extract(self, stim):
        rmse = librosa.feature.rmse(y=stim.data,
                                    frame_length=self.frame_length,
                                    hop_length=self.hop_length,
                                    center=self.center,
                                    pad_mode=self.pad_mode)[0]
        n_frames = len(rmse)
        onsets = librosa.frames_to_time(range(n_frames),
                                        sr=stim.sampling_rate,
                                        hop_length=self.hop_length)
        durations = [self.hop_length / float(stim.sampling_rate)] * n_frames
        return ExtractorResult(rmse, stim, self,
                               features=['RMSE'],
                               onsets=onsets,
                               durations=durations)


class ZeroCrossingRateExtractor(LibrosaFeatureExtractor):

    ''' Extracts the zero-crossing rate over time frames of audio. '''

    _log_attributes = ('frame_length', 'hop_length', 'center')

    def __init__(self, frame_length=2048, hop_length=512, center=True):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.center = center
        super(ZeroCrossingRateExtractor, self).__init__()

    def _extract(self, stim):
        zcr = librosa.feature.zero_crossing_rate(stim.data,
                                                 frame_length=self.frame_length,
                                                 hop_length=self.hop_length,
                                                 center=self.center)[0]
        n_frames = len(zcr)
        onsets = librosa.frames_to_time(range(n_frames),
                                        sr=stim.sampling_rate,
                                        hop_length=self.hop_length)
        durations = [self.hop_length / float(stim.sampling_rate)] * n_frames
        return ExtractorResult(zcr, stim, self,
                               features=['zero_crossing_rate'],
                               onsets=onsets,
                               durations=durations)


class ChromaSTFTExtractor(LibrosaFeatureExtractor):

    ''' '''

    def __init__(self):
        super(ChromaSTFTExtractor, self).__init__()

    def _extract(self, stim):
        pass


class MFCCExtractor(LibrosaFeatureExtractor):

    ''' '''

    def __init__(self):
        super(MFCCExtractor, self).__init__()

    def _extract(self, stim):
        pass
