from featurex import stimuli
from featurex.extractors import StimExtractor
import numpy as np
from scipy import fft
from featurex.core import Value, Event


class AudioExtractor(StimExtractor):

    target = stimuli.audio.AudioStim


class STFTExtractor(AudioExtractor):
    ''' Short-time Fourier Transform extractor. '''
    def __init__(self, frame_size=0.5, hop_size=0.1, bins=5,
                 spectrogram=False):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.spectrogram = spectrogram
        self.freq_bins = bins

    def stft(self, stim):
        ''' code adapted from http://stackoverflow.com/questions/2459295/invertible-stft-and-istft-in-python '''
        x = stim.data
        framesamp = int(self.frame_size*stim.sampling_rate)
        hopsamp = int(self.hop_size*stim.sampling_rate)
        w = np.hanning(framesamp)
        X = np.array([fft(w*x[i:(i+framesamp)]) for i in range(0, len(x)-framesamp, hopsamp)])
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

    def apply(self, stim):
        data = self.stft(stim)
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
