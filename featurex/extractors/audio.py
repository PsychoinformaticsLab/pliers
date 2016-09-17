from featurex.stimuli import audio
from featurex.extractors import Extractor
import numpy as np
from scipy import fft
from featurex.core import Value, Event


class AudioExtractor(Extractor):

    ''' Base Audio Extractor class; all subclasses can only be applied to
    audio. '''
    target = audio.AudioStim


class TranscribedAudioExtractor(AudioExtractor):
    
    ''' Base Transcribed Audio Extractor class; all subclasses can only be
    applied to transcribed audio. '''
    target = audio.TranscribedAudioStim


class STFTExtractor(AudioExtractor):

    ''' Short-time Fourier Transform extractor.
    Args:
        frame_size (float): The width of the frame/window to apply an FFT to,
            in seconds.
        hop_size (float): The step size to increment the window by on each
            iteration, in seconds (effectively, the sampling rate).
        bins (list or int): The set of bins or frequency bands to extract
            power for. If an int is passed, this is the number of bins
            returned, with each bin spanning an equal range of frequencies.
            E.g., if bins=5 and the frequency spectrum runs from 0 to 20KHz,
            each bin will span 4KHz. If a list is passed, each element must be
            a tuple or list of lower and upper frequency bounds. E.g., passing
            [(0, 300), (300, 3000)] would compute power in two bands, one
            between 0 and 300Hz, and one between 300Hz and 3KHz.
        spectrogram (bool): If True, plots a spectrogram of the results.

    Notes: code adapted from http://stackoverflow.com/questions/2459295/invertible-stft-and-istft-in-python
    '''

    def __init__(self, frame_size=0.5, hop_size=0.1, bins=5,
                 spectrogram=False):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.spectrogram = spectrogram
        self.freq_bins = bins
        super(STFTExtractor, self).__init__()

    def _stft(self, stim):
        x = stim.data
        framesamp = int(self.frame_size*stim.sampling_rate)
        hopsamp = int(self.hop_size*stim.sampling_rate)
        w = np.hanning(framesamp)
        X = np.array([fft(w*x[i:(i+framesamp)])
                      for i in range(0, len(x)-framesamp, hopsamp)])
        nyquist_lim = X.shape[1]/2
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

    def _transform(self, stim):
        data = self._stft(stim)
        events = []
        time_bins = np.arange(0., stim.duration-self.frame_size, self.hop_size)

        if isinstance(self.freq_bins, int):
            bins = []
            bin_size = data.shape[1] / self.freq_bins
            for i in range(self.freq_bins):
                bins.append((i*bin_size, (i+1)*bin_size))
            self.freq_bins = bins

        for i, tb in enumerate(time_bins):
            ev = Event(onset=tb, duration=self.frame_size)
            value_data = {}
            for fb in self.freq_bins:
                label = '%d_%d' % fb
                start, stop = fb
                val = data[i, start:stop].mean()
                if np.isinf(val):
                    val = 0.
                value_data[label] = val
            ev.add_value(Value(stim, self, value_data))
            events.append(ev)
        return events


class MeanAmplitudeExtractor(TranscribedAudioExtractor):
    
    ''' Mean amplitude extractor for blocks of audio with transcription.
    '''
    
    def __init__(self):
        pass
    
    def _transform(self, stim):
        amps = stim.data
        sampling_rate = stim.sampling_rate
        elements = stim.transcription.elements
        events = []
        for i, el in enumerate(elements):
            onset = sampling_rate * el.onset
            duration = sampling_rate * el.duration
            
            r_onset = np.round(onset).astype(int)
            r_offset = np.round(onset+duration).astype(int)
            if not r_offset <= amps.shape[0]:
                raise Exception('Block ends after data.')
            
            mean_amplitude = np.mean(amps[r_onset:r_offset])
            amplitude_data = {'mean_amplitude': mean_amplitude}
            ev = Event(onset=onset, duration=duration)
            ev.add_value(Value(stim, self, amplitude_data))
            events.append(ev)
        return events
