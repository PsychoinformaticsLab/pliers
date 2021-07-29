''' Extractors that operate on AudioStim inputs. '''

from abc import ABCMeta
from os import path
import sys
import logging

import numpy as np
from scipy import fft
import pandas as pd

from pliers.stimuli.audio import AudioStim
from pliers.stimuli.text import ComplexTextStim
from pliers.extractors.base import Extractor, ExtractorResult
from pliers.utils import attempt_to_import, verify_dependencies, listify
from pliers.support.exceptions import MissingDependencyError
from pliers.support.setup_yamnet import YAMNET_PATH

librosa = attempt_to_import('librosa')
tf = attempt_to_import('tensorflow')


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
    VERSION = '1.0'

    def __init__(self, frame_size=0.5, hop_size=0.1, freq_bins=5,
                 spectrogram=False):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.spectrogram = spectrogram
        self.freq_bins = freq_bins
        super().__init__()

    def _stft(self, stim):
        x = stim.data
        framesamp = int(self.frame_size * stim.sampling_rate)
        hopsamp = int(self.hop_size * stim.sampling_rate)
        w = np.hanning(framesamp)
        X = np.array([fft.fft(w * x[i:(i + framesamp)])
                      for i in range(0, len(x) - framesamp, hopsamp)])
        nyquist_lim = int(X.shape[1] // 2)
        X = np.log(X[:, :nyquist_lim])
        X = np.absolute(X)
        if self.spectrogram:
            import matplotlib.pyplot as plt
            bins = np.fft.fftfreq(framesamp, d=1. / stim.sampling_rate)
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
        time_bins = np.arange(0., stim.duration - self.frame_size,
                              self.hop_size)

        if isinstance(self.freq_bins, int):
            bins = []
            bin_size = int(data.shape[1] / self.freq_bins)
            for i in range(self.freq_bins):
                if i == self.freq_bins - 1:
                    bins.append((i * bin_size, data.shape[1]))
                else:
                    bins.append((i * bin_size, (i + 1) * bin_size))
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
                               onsets=index, durations=self.hop_size,
                               orders=list(range(len(index))))


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
            r_offset = np.round(onset + duration).astype(int)
            if not r_offset <= amps.shape[0]:
                raise Exception('Block ends after data.')

            mean_amplitude = np.mean(amps[r_onset:r_offset])
            values.append(mean_amplitude)

        orders = list(range(len(elements)))

        return ExtractorResult(values, stim, self, features=['mean_amplitude'],
                               onsets=onsets, durations=durations,
                               orders=orders)


class LibrosaFeatureExtractor(AudioExtractor, metaclass=ABCMeta):

    ''' A generic class for audio extractors using the librosa library. '''

    _log_attributes = ('hop_length', 'librosa_kwargs')

    def __init__(self, feature=None, hop_length=512, **librosa_kwargs):
        verify_dependencies(['librosa'])
        if feature:
            self._feature = feature
        self.hop_length = hop_length
        self.librosa_kwargs = librosa_kwargs
        super().__init__()

    def get_feature_names(self):
        return self._feature

    def _get_values(self, stim):
        if self._feature in ['zero_crossing_rate', 'rms', 'spectral_flatness']:
            return getattr(librosa.feature, self._feature)(
                y=stim.data, hop_length=self.hop_length, **self.librosa_kwargs)
        elif self._feature == 'tonnetz':
            return getattr(librosa.feature, self._feature)(
                y=stim.data, sr=stim.sampling_rate, **self.librosa_kwargs)

        elif self._feature in[ 'onset_detect', 'onset_strength_multi']:
            return getattr(librosa.onset, self._feature)(
                y=stim.data, sr=stim.sampling_rate, hop_length=self.hop_length,
                **self.librosa_kwargs)

        elif self._feature in[ 'tempo', 'beat_track']:
            return getattr(librosa.beat, self._feature)(
                y=stim.data, sr=stim.sampling_rate, hop_length=self.hop_length,
                **self.librosa_kwargs)

        elif self._feature in[ 'harmonic', 'percussive']:
            return getattr(librosa.effects, self._feature)(
                y=stim.data,
                **self.librosa_kwargs)
        else:
            return getattr(librosa.feature, self._feature)(
                y=stim.data, sr=stim.sampling_rate, hop_length=self.hop_length,
                **self.librosa_kwargs)

    def _extract(self, stim):

        values = self._get_values(stim)

        if self._feature=='beat_track':
            beats=np.array(values[1])
            values=beats

        values = values.T
        n_frames = len(values)

        feature_names = listify(self.get_feature_names())

        onsets = librosa.frames_to_time(range(n_frames),
                                        sr=stim.sampling_rate,
                                        hop_length=self.hop_length)

        onsets = onsets + stim.onset if stim.onset else onsets

        durations = [self.hop_length / float(stim.sampling_rate)] * n_frames

        return ExtractorResult(values, stim, self, features=feature_names,
                               onsets=onsets, durations=durations,
                               orders=list(range(n_frames)))


class SpectralCentroidExtractor(LibrosaFeatureExtractor):

    ''' Extracts the spectral centroids from audio using the Librosa library.

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.feature.spectral_centroid.html.'''

    _feature = 'spectral_centroid'


class SpectralBandwidthExtractor(LibrosaFeatureExtractor):

    ''' Extracts the p'th-order spectral bandwidth from audio using the
    Librosa library.

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.feature.spectral_bandwidth.html.'''

    _feature = 'spectral_bandwidth'


class SpectralFlatnessExtractor(LibrosaFeatureExtractor):

    ''' Computes the spectral flatness from audio using the
    Librosa library.

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.feature.spectral_flatness.html.'''

    _feature = 'spectral_flatness'


class SpectralContrastExtractor(LibrosaFeatureExtractor):

    ''' Extracts the spectral contrast from audio using the Librosa library.

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.feature.spectral_contrast.html.'''

    _feature = 'spectral_contrast'

    def __init__(self, n_bands=6, **kwargs):
        self.n_bands = n_bands
        super().__init__(
            n_bands=n_bands, **kwargs)

    def get_feature_names(self):
        abc= ['spectral_contrast_band_%d' % i
                for i in range(self.n_bands + 1)]
        return abc


class SpectralRolloffExtractor(LibrosaFeatureExtractor):

    ''' Extracts the roll-off frequency from audio using the Librosa library.

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.feature.spectral_rolloff.html.'''

    _feature = 'spectral_rolloff'


class PolyFeaturesExtractor(LibrosaFeatureExtractor):

    ''' Extracts the coefficients of fitting an nth-order polynomial to the columns of an audio's spectrogram (via Librosa).

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.feature.poly_features.html.'''

    _feature = 'poly_features'

    def __init__(self, order=1, **kwargs):
        self.order = order
        super().__init__(order=order, **kwargs)

    def get_feature_names(self):
        return ['coefficient_%d' % i for i in range(self.order + 1)]


class RMSExtractor(LibrosaFeatureExtractor):

    ''' Extracts root mean square (RMS) from audio using the Librosa
    library.

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.feature.rms.html.'''

    _feature = 'rms'


class OnsetDetectExtractor(LibrosaFeatureExtractor):

    ''' Detects the basic onset (onset_detect) from audio using the Librosa
    library.

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.onset.onset_detect.html.'''

    _feature = 'onset_detect'


class TempoExtractor(LibrosaFeatureExtractor):

    ''' Detects the tempo (tempo) from audio using the Librosa
    library.

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.beat.tempo.html.'''

    _feature = 'tempo'


class BeatTrackExtractor(LibrosaFeatureExtractor):

    ''' Dynamic programming beat tracker (beat_track) from audio using the Librosa
    library.

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.beat.beat_track.html.'''

    _feature = 'beat_track'


class OnsetStrengthMultiExtractor(LibrosaFeatureExtractor):

    '''Computes the spectral flux onset strength envelope across multiple channels (onset_strength_multi) from audio using the Librosa
    library.

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.onset.onset_strength_multi.html.'''

    _feature = 'onset_strength_multi'


class ZeroCrossingRateExtractor(LibrosaFeatureExtractor):

    ''' Extracts the zero-crossing rate of audio using the Librosa library.

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.feature.zero_crossing_rate.html.'''

    _feature = 'zero_crossing_rate'


class ChromaSTFTExtractor(LibrosaFeatureExtractor):

    ''' Extracts a chromagram from an audio's waveform using the Librosa
    library.

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.feature.chroma_stft.html.'''

    _feature = 'chroma_stft'

    def __init__(self, n_chroma=12, **kwargs):
        self.n_chroma = n_chroma
        super().__init__(n_chroma=n_chroma, **kwargs)

    def get_feature_names(self):
        return ['chroma_%d' % i for i in range(self.n_chroma)]


class ChromaCQTExtractor(LibrosaFeatureExtractor):

    ''' Extracts a constant-q chromogram from audio using the Librosa library.

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.feature.chroma_cqt.html.'''

    _feature = 'chroma_cqt'

    def __init__(self, n_chroma=12, **kwargs):
        self.n_chroma = n_chroma
        super().__init__(n_chroma=n_chroma, **kwargs)

    def get_feature_names(self):
        return ['chroma_cqt_%d' % i for i in range(self.n_chroma)]


class ChromaCENSExtractor(LibrosaFeatureExtractor):

    ''' Extracts a chroma variant "Chroma Energy Normalized" (CENS)
    chromogram from audio (via Librosa).

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.feature.chroma_cens.html.'''

    _feature = 'chroma_cens'

    def __init__(self, n_chroma=12, **kwargs):
        self.n_chroma = n_chroma
        super().__init__(n_chroma=n_chroma, **kwargs)

    def get_feature_names(self):
        return ['chroma_cens_%d' % i for i in range(self.n_chroma)]


class MelspectrogramExtractor(LibrosaFeatureExtractor):

    ''' Extracts mel-scaled spectrogram from audio using the Librosa library.

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html.'''

    _feature = 'melspectrogram'

    def __init__(self, n_mels=128, **kwargs):
        self.n_mels = n_mels
        super().__init__(n_mels=n_mels, **kwargs)

    def get_feature_names(self):
        return ['mel_%d' % i for i in range(self.n_mels)]


class MFCCExtractor(LibrosaFeatureExtractor):

    ''' Extracts Mel Frequency Ceptral Coefficients from audio using the
    Librosa library.

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html.'''

    _feature = 'mfcc'

    def __init__(self, n_mfcc=20, **kwargs):
        self.n_mfcc = n_mfcc
        super().__init__(n_mfcc=n_mfcc, **kwargs)

    def get_feature_names(self):
        return ['mfcc_%d' % i for i in range(self.n_mfcc)]


class TonnetzExtractor(LibrosaFeatureExtractor):

    ''' Extracts the tonal centroids (tonnetz) from audio using the Librosa
    library.

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.feature.tonnetz.html.'''

    _feature = 'tonnetz'

    def get_feature_names(self):
        return ['tonal_centroid_%d' % i for i in range(6)]


class TempogramExtractor(LibrosaFeatureExtractor):

    ''' Extracts a tempogram from audio using the Librosa library.

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.feature.tempogram.html.'''

    _feature = 'tempogram'

    def __init__(self, win_length=384, **kwargs):
        self.win_length = win_length
        super().__init__(win_length=win_length,
                                                 **kwargs)

    def get_feature_names(self):
        return ['tempo_%d' % i for i in range(self.win_length)]


class HarmonicExtractor(LibrosaFeatureExtractor):

    ''' Extracts the harmonic elements from an audio time-series using the Librosa library.

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.effects.harmonic.html.'''

    _feature = 'harmonic'


class PercussiveExtractor(LibrosaFeatureExtractor):

    ''' Extracts the percussive elements from an audio time-series using the Librosa library.

    For details on argument specification visit:
    https://librosa.org/doc/latest/generated/librosa.effects.percussive.html.'''

    _feature = 'percussive'


class AudiosetLabelExtractor(AudioExtractor):

    ''' Extract probability of 521 audio event classes based on AudioSet
    corpus using a YAMNet architecture. Code available at:
    https://github.com/tensorflow/models/tree/master/research/audioset/yamnet

    Args:
    hop_size (float): size of the audio segment (in seconds) on which label
        extraction is performed.
    top_n (int): specifies how many of the highest label probabilities are
        returned. If None, all labels (or all in labels) are returned.
        Top_n and labels are mutually exclusive arguments.
    labels (list): specifies subset of labels for which probabilities
        are to be returned. If None, all labels (or top_n) are returned.
        The full list of labels is available in the audioset/yamnet
        repository (see yamnet_class_map.csv).
    weights_path (optional): full path to model weights file. If not provided,
        weights from pretrained YAMNet module are used.
    yamnet_kwargs (optional): Optional named arguments that modify input
        parameters for the model (see params.py file in yamnet repository)
    '''

    _log_attributes = ('hop_size', 'top_n', 'labels', 'weights_path',
                       'yamnet_kwargs')

    def __init__(self, hop_size=0.1, top_n=None, labels=None,
                 weights_path=None, **yamnet_kwargs):
        verify_dependencies(['tensorflow'])
        try:
            sys.path.insert(0, str(YAMNET_PATH))
            self.yamnet = attempt_to_import('yamnet')
            verify_dependencies(['yamnet'])
        except MissingDependencyError:
            msg = ('Yamnet could not be imported. To download and set up '
                  'yamnet, run:\n\tpython -m pliers.support.setup_yamnet')
            raise MissingDependencyError(dependencies=None,
                                         custom_message=msg)
        if top_n and labels:
            raise ValueError('Top_n and labels are mutually exclusive '
                             'arguments. Reinstantiate the extractor setting '
                             'top_n or labels to None (or leaving it '
                             'unspecified).')

        MODULE_PATH = path.dirname(self.yamnet.__file__)
        LABELS_PATH = path.join(MODULE_PATH, 'yamnet_class_map.csv')
        self.weights_path = weights_path or path.join(MODULE_PATH, 'yamnet.h5')
        self.hop_size = hop_size
        self.yamnet_kwargs = yamnet_kwargs or {}
        self.params = self.yamnet.params
        self.params.PATCH_HOP_SECONDS = hop_size
        for par, v in self.yamnet_kwargs.items():
            setattr(self.params, par, v)
        if self.params.PATCH_WINDOW_SECONDS != 0.96:
            logging.warning('Custom values for PATCH_WINDOW_SECONDS were '
                'passed. YAMNet was trained on windows of 0.96s. Different '
                'values might yield unreliable results.')

        self.top_n = top_n
        all_labels = pd.read_csv(LABELS_PATH)['display_name'].tolist()
        if labels is not None:
            missing = list(set(labels) - set(all_labels))
            labels = list(set(labels) & set(all_labels))
            if missing:
                logging.warning(f'Labels {missing} do not exist. Dropping.')
            self.label_idx, self.labels = zip(*[(i,l)
                                               for i,l in enumerate(all_labels)
                                               if l in labels])
        else:
            self.labels = all_labels
            self.label_idx = list(range(len(all_labels)))
        super().__init__()

    def _extract(self, stim):
        self.params.SAMPLE_RATE = stim.sampling_rate

        if self.params.SAMPLE_RATE >= 2 * self.params.MEL_MAX_HZ:
            if self.params.SAMPLE_RATE != 16000:
                logging.warning(
                    'The sampling rate of the stimulus is '
                    f'{self.params.SAMPLE_RATE}Hz. YAMNet was trained on '
                    ' audio sampled at 16000Hz. This should not impact '
                    'predictions, but you can resample the input using '
                    'AudioResamplingFilter for full conformity '
                    'to training.')
            if self.params.MEL_MIN_HZ != 125 or self.params.MEL_MAX_HZ != 7500:
                logging.warning(
                    'Custom values for MEL_MIN_HZ and MEL_MAX_HZ '
                    'were passed. Changing these defaults might affect '
                    'model performance.')
        else:
            raise ValueError(
                'The sampling rate of your stimulus '
                f'({self.params.SAMPLE_RATE}Hz) must be at least twice the '
                f'value of MEL_MAX_HZ ({self.params.MEL_MAX_HZ}Hz). Upsample'
                ' your audio stimulus (recommended) or pass a lower value of '
                'MEL_MAX_HZ when initializing the extractor.')

        model = self.yamnet.yamnet_frames_model(self.params)
        model.load_weights(self.weights_path)
        preds, _ = model.predict_on_batch(np.reshape(stim.data, [1,-1]))
        preds = preds[:,self.label_idx]

        nr_lab = self.top_n or len(self.labels)
        idx = np.mean(preds,axis=0).argsort()
        preds = np.fliplr(preds[:,idx][:,-nr_lab:])
        labels = [self.labels[i] for i in idx][-nr_lab:][::-1]

        hop = self.params.PATCH_HOP_SECONDS
        window = self.params.PATCH_WINDOW_SECONDS
        stft_window = self.params.STFT_WINDOW_SECONDS
        stft_hop = self.params.STFT_HOP_SECONDS
        dur = window + stft_window - stft_hop
        onsets = np.arange(start=0, stop=stim.duration - dur, step=hop)

        return ExtractorResult(preds, stim, self, features=labels,
                               onsets=onsets, durations=[dur]*len(onsets),
                               orders=list(range(len(onsets))))



class MFCCEnergyExtractor(MFCCExtractor):
    ''' Given a Middle Bound, Extract the Energy from the Low/High Register
    of Mel-Frequency Ceptral Coefficients.

    Args:
    n_mfcc (int): specifies the number of MFCC to extract
    n_coefs(int): cepstrum coefficients to keep in the high/low quefrency spectrum
    hop_length (int): hop length in number of samples
    librosa_kwargs (optional): Optional named arguments to pass to librosa
    '''

    _log_attributes = ('hop_length', 'n_coefs', 'n_mfcc')

    def __init__(self, n_mfcc=48, n_coefs=13, hop_length=1024, 
                 direction='low', **librosa_kwargs):
        assert direction in ['low', 'high']
        self.n_mfcc = n_mfcc
        self.n_coefs = n_coefs
        self.hop_length = hop_length
        self.librosa_kwargs = librosa_kwargs
        self.direction = direction
        super().__init__(n_mfcc=n_mfcc)

    def _get_values(self,stim):
        vals = super()._get_values(stim)
        subset = vals[:self.n_coefs] if self.direction == 'low' \
            else vals[self.n_coefs:]
            
        assert 0
        subset_mfs = fft.dct(np.transpose(subset), 
            type=2, 
            n=self.n_mfcc, 
            axis=- 1, 
            norm='ortho', 
            overwrite_x=False)
        subset_mfs = 10 ** (subset_mfs / 20.)
        return subset_mfs.T

    def get_feature_names(self):
        return ['mfcc_energy_%d' % i for i in range(self.n_mfcc)]