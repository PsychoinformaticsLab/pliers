from os.path import join
from os import environ

import numpy as np
import pytest

from ..utils import get_test_data_path, SemVerDict
from pliers.extractors import (LibrosaFeatureExtractor,
                               STFTAudioExtractor,
                               MeanAmplitudeExtractor,
                               SpectralCentroidExtractor,
                               SpectralBandwidthExtractor,
                               SpectralContrastExtractor,
                               SpectralRolloffExtractor,
                               PolyFeaturesExtractor,
                               ZeroCrossingRateExtractor,
                               ChromaSTFTExtractor,
                               ChromaCQTExtractor,
                               ChromaCENSExtractor,
                               MelspectrogramExtractor,
                               MFCCExtractor,
                               TonnetzExtractor,
                               TempogramExtractor,
                               RMSExtractor,
                               SpectralFlatnessExtractor,
                               OnsetDetectExtractor,
                               OnsetStrengthMultiExtractor,
                               TempoExtractor,
                               BeatTrackExtractor,
                               HarmonicExtractor,
                               PercussiveExtractor,
                               AudiosetLabelExtractor,
                               MFCCEnergyExtractor)
from pliers.stimuli import (ComplexTextStim, AudioStim,
                            TranscribedAudioCompoundStim)
from pliers.filters import AudioResamplingFilter
from pliers.utils import attempt_to_import, verify_dependencies

from librosa import __version__ as LIBROSA_VERSION

AUDIO_DIR = join(get_test_data_path(), 'audio')
tf = attempt_to_import('tensorflow')

def test_stft_extractor():
    stim = AudioStim(join(AUDIO_DIR, 'barber.wav'), onset=4.2)
    ext = STFTAudioExtractor(frame_size=1., spectrogram=False,
                             freq_bins=[(100, 300), (300, 3000),
                                        (3000, 20000)])
    result = ext.transform(stim)
    df = result.to_df()
    assert df.shape == (557, 7)
    assert df['onset'][0] == 4.2

    ext = STFTAudioExtractor(frame_size=1., spectrogram=False,
                             freq_bins=5)
    result = ext.transform(stim)
    df = result.to_df(timing=False, object_id=False)
    assert df.shape == (557, 5)
    assert '0_1102' in df.columns


def test_mean_amplitude_extractor():
    audio = AudioStim(join(AUDIO_DIR, 'barber_edited.wav'))
    text_file = join(get_test_data_path(), 'text', 'wonderful_edited.srt')
    text = ComplexTextStim(text_file)
    stim = TranscribedAudioCompoundStim(audio=audio, text=text)
    ext = MeanAmplitudeExtractor()
    result = ext.transform(stim).to_df()
    targets = [-0.154661, 0.121521]
    assert np.allclose(result['mean_amplitude'], targets)


def test_librosa_extractor():
    audio = AudioStim(join(AUDIO_DIR, 'barber.wav'))
    ext = LibrosaFeatureExtractor(feature='rms')
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 5)
    assert np.isclose(df['onset'][1], 0.04644)
    assert np.isclose(df['duration'][0], 0.04644)
    assert np.isclose(df['rms'][0], 0.25663)


def test_spectral_extractors():
    audio = AudioStim(join(AUDIO_DIR, 'barber.wav'))
    ext = SpectralCentroidExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 5)
    assert np.isclose(df['onset'][1], 0.04644)
    assert np.isclose(df['duration'][0], 0.04644)
    assert np.isclose(df['spectral_centroid'][0], 1144.98145)

    ext2 = SpectralCentroidExtractor(n_fft=1024, hop_length=256)
    df = ext2.transform(audio).to_df()
    assert df.shape == (2441, 5)
    assert np.isclose(df['onset'][1], 0.02322)
    assert np.isclose(df['duration'][0], 0.02322)
    assert np.isclose(df['spectral_centroid'][0], 866.20176)

    ext = SpectralBandwidthExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 5)
    assert np.isclose(df['spectral_bandwidth'][0], 1172.96090)

    ext = SpectralContrastExtractor(fmin=100.0)
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 11)
    assert np.isclose(df['spectral_contrast_band_4'][0], 25.637166)

    ext = SpectralRolloffExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 5)
    assert np.isclose(df['spectral_rolloff'][0], 2492.46826)


def test_polyfeatures_extractor():
    audio = AudioStim(join(AUDIO_DIR, 'barber.wav'))
    ext = PolyFeaturesExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 6)
    assert np.isclose(df['onset'][1], 0.04644)
    assert np.isclose(df['duration'][0], 0.04644)
    assert np.isclose(df['coefficient_0'][0], -0.00172077)

    ext2 = PolyFeaturesExtractor(order=3)
    df = ext2.transform(audio).to_df()
    assert df.shape == (1221, 8)
    assert np.isclose(df['coefficient_3'][2], 12.32108)


def test_zcr_extractor():
    audio = AudioStim(join(AUDIO_DIR, 'barber.wav'),
                      onset=2.0)
    ext = ZeroCrossingRateExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 5)
    assert np.isclose(df['onset'][1], 2.04644)
    assert np.isclose(df['duration'][0], 0.04644)
    assert np.isclose(df['zero_crossing_rate'][0], 0.069824)

    ext2 = ZeroCrossingRateExtractor(frame_length=1024, hop_length=256,
                                     center=False, pad=True)
    df = ext2.transform(audio).to_df()
    assert df.shape == (2437, 5)
    assert np.isclose(df['onset'][1], 2.02322)
    assert np.isclose(df['duration'][0], 0.02322)
    assert np.isclose(df['zero_crossing_rate'][0], 0.140625)


def test_chroma_extractors():
    answers = SemVerDict(
        {
            '>=0.6.3,<0.8.0': {'cqt': 0.336481, 'cens': 0.136409},
            '>=0.8.0':        {'cqt': 0.508431, 'cens': 0.102370}
        }
    )

    audio = AudioStim(join(AUDIO_DIR, 'barber.wav'))
    ext = ChromaSTFTExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 16)
    assert np.isclose(df['onset'][1], 0.04644)
    assert np.isclose(df['duration'][0], 0.04644)
    assert np.isclose(df['chroma_2'][0], 0.53129)

    ext2 = ChromaSTFTExtractor(n_chroma=6, n_fft=1024, hop_length=256)
    df = ext2.transform(audio).to_df()
    assert df.shape == (2441, 10)
    assert np.isclose(df['onset'][1], 0.02322)
    assert np.isclose(df['duration'][0], 0.02322)
    assert np.isclose(df['chroma_5'][0], 0.42069)

    ext = ChromaCQTExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 16)
    assert np.isclose(df['chroma_cqt_2'][0], answers[LIBROSA_VERSION]['cqt'])

    ext = ChromaCENSExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 16)
    assert np.isclose(df['chroma_cens_2'][0], answers[LIBROSA_VERSION]['cens'])


def test_melspectrogram_extractor():
    audio = AudioStim(join(AUDIO_DIR, 'barber.wav'))
    ext = MelspectrogramExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 132)
    assert np.isclose(df['onset'][1], 0.04644)
    assert np.isclose(df['duration'][0], 0.04644)
    assert np.isclose(df['mel_3'][0], 0.82194)

    ext2 = MelspectrogramExtractor(n_mels=15)
    df = ext2.transform(audio).to_df()
    assert df.shape == (1221, 19)
    assert np.isclose(df['mel_4'][2], 7.40387)


def test_mfcc_extractor():
    audio = AudioStim(join(AUDIO_DIR, 'barber.wav'))
    ext = MFCCExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 24)
    assert np.isclose(df['onset'][1], 0.04644)
    assert np.isclose(df['duration'][0], 0.04644)
    assert np.isclose(df['mfcc_3'][0], 20.84870)

    ext2 = MFCCExtractor(n_mfcc=15)
    df = ext2.transform(audio).to_df()
    assert df.shape == (1221, 19)
    assert np.isclose(df['mfcc_14'][2], -22.39406)


def test_tonnetz_extractor():
    audio = AudioStim(join(AUDIO_DIR, 'barber.wav'))
    ext = TonnetzExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 10)
    assert np.isclose(df['onset'][1], 0.04644)
    assert np.isclose(df['duration'][0], 0.04644)

    tonal_centroid_answers = SemVerDict({'>=0.6.3,<0.8.0': -0.0391266, '>=0.8.0': -0.0804008})
    assert np.isclose(df['tonal_centroid_0'][0], tonal_centroid_answers[LIBROSA_VERSION])


def test_tempogram_extractor():
    audio = AudioStim(join(AUDIO_DIR, 'barber.wav'))
    ext = TempogramExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 388)
    assert np.isclose(df['onset'][1], 0.04644)
    assert np.isclose(df['duration'][0], 0.04644)
    assert np.isclose(df['tempo_1'][0], 0.75708)

    ext2 = TempogramExtractor(win_length=300)
    df = ext2.transform(audio).to_df()
    assert df.shape == (1221, 304)
    assert np.isclose(df['tempo_1'][2], 0.74917)


def test_rms_extractor():
    audio = AudioStim(join(AUDIO_DIR, 'barber.wav'))
    ext = RMSExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 5)
    assert np.isclose(df['onset'][2], 0.092880)
    assert np.isclose(df['duration'][2], 0.04644)
    assert np.isclose(df['rms'][2], 0.229993)

    assert np.isclose(df['onset'][4], 0.185760)
    assert np.isclose(df['duration'][4], 0.04644)
    assert np.isclose(df['rms'][4], 0.184349)

    assert np.isclose(df['onset'][1219], 56.610249)
    assert np.isclose(df['duration'][1219], 0.04644)
    assert np.isclose(df['rms'][1219], 0.001348, rtol=1e-03)


def test_spectral_flatness_extractor():
    audio = AudioStim(join(AUDIO_DIR, 'barber.wav'))
    ext = SpectralFlatnessExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 5)
    assert np.isclose(df['onset'][10], 0.464399)
    assert np.isclose(df['duration'][10], 0.04644)
    assert np.isclose(df['spectral_flatness'][10], 0.035414)

    assert np.isclose(df['onset'][25], 1.160998)
    assert np.isclose(df['duration'][25], 0.04644)
    assert np.isclose(df['spectral_flatness'][25], 0.084409)

    assert np.isclose(df['onset'][1215], 56.424490)
    assert np.isclose(df['duration'][1215], 0.04644)
    assert np.isclose(df['spectral_flatness'][1215], 0.0349364)


def test_onset_detect_extractor():
    audio = AudioStim(join(AUDIO_DIR, 'barber.wav'))
    ext = OnsetDetectExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (150, 5)
    assert np.isclose(df['onset'][1], 0.046440)
    assert np.isclose(df['duration'][1], 0.04644)
    assert np.isclose(df['onset_detect'][1], 56)

    assert np.isclose(df['onset'][5], 0.232200)
    assert np.isclose(df['duration'][5], 0.04644)
    assert np.isclose(df['onset_detect'][5], 85)

    assert np.isclose(df['onset'][121], 5.619229)
    assert np.isclose(df['duration'][121], 0.04644)
    assert np.isclose(df['onset_detect'][121], 896)


def test_onset_multi_extractor():
    audio = AudioStim(join(AUDIO_DIR, 'barber.wav'))
    ext = OnsetStrengthMultiExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 5)
    assert np.isclose(df['onset'][10], 0.464399)
    assert np.isclose(df['duration'][10], 0.04644)
    assert np.isclose(df['onset_strength_multi'][10], 0.821330)

    assert np.isclose(df['onset'][15], 0.696599)
    assert np.isclose(df['duration'][15], 0.04644)
    assert np.isclose(df['onset_strength_multi'][15], 0.430218)

    assert np.isclose(df['onset'][1218], 56.563810)
    assert np.isclose(df['duration'][1218], 0.04644)
    assert np.isclose(df['onset_strength_multi'][1218], 0.058071)


def test_tempo_extractor():
    audio = AudioStim(join(AUDIO_DIR, 'barber.wav'))
    ext = TempoExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (1, 5)
    assert np.isclose(df['onset'][0], 0.0)
    assert np.isclose(df['duration'][0], 0.04644)
    assert np.isclose(df['tempo'][0], 117.453835)


def test_beat_track_extractor():
    audio = AudioStim(join(AUDIO_DIR, 'barber.wav'))
    ext = BeatTrackExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (102, 5)
    assert np.isclose(df['onset'][2], 0.092880)
    assert np.isclose(df['duration'][2], 0.04644)
    assert np.isclose(df['beat_track'][2], 87)

    assert np.isclose(df['onset'][29], 1.346757)
    assert np.isclose(df['duration'][29], 0.04644)
    assert np.isclose(df['beat_track'][29], 389)

    assert np.isclose(df['onset'][101], 4.690431)
    assert np.isclose(df['duration'][101], 0.04644)
    assert np.isclose(df['beat_track'][101], 1195)


def test_harmonic_extractor():
    audio = AudioStim(join(AUDIO_DIR, 'barber.wav'))
    ext = HarmonicExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (624786, 5)
    assert np.isclose(df['onset'][8], 0.371519)
    assert np.isclose(df['duration'][8], 0.04644)
    assert np.isclose(df['harmonic'][8], -0.026663, rtol=1e-4)

    assert np.isclose(df['onset'][19], 0.882358)
    assert np.isclose(df['duration'][19], 0.04644)
    assert np.isclose(df['harmonic'][19], 0.031422, rtol=1e-4)

    assert np.isclose(df['onset'][29], 1.346757)
    assert np.isclose(df['duration'][29], 0.04644)
    assert np.isclose(df['harmonic'][29], -0.004497, rtol=1e-4)


def test_percussion_extractor():
    audio = AudioStim(join(AUDIO_DIR, 'barber.wav'))
    ext = PercussiveExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (624786, 5)
    assert np.isclose(df['onset'][9], 0.417959)
    assert np.isclose(df['duration'][9], 0.04644)
    assert np.isclose(df['percussive'][9], 0.028902, rtol=1e-4)

    assert np.isclose(df['onset'][17], 0.789478)
    assert np.isclose(df['duration'][17], 0.04644)
    assert np.isclose(df['percussive'][17], -0.031428, rtol=1e-4)

    assert np.isclose(df['onset'][29], 1.346757)
    assert np.isclose(df['duration'][29], 0.04644)
    assert np.isclose(df['percussive'][29], 0.004497, rtol=1e-4)

@pytest.mark.parametrize('hop_size', [0.1, 1])
@pytest.mark.parametrize('top_n', [5, 10])
@pytest.mark.parametrize('target_sr', [22000, 14000])
def test_audioset_extractor(hop_size, top_n, target_sr):
    verify_dependencies(['tensorflow'])

    def compute_expected_length(stim, ext):
        stft_par = ext.params.STFT_WINDOW_SECONDS - ext.params.STFT_HOP_SECONDS
        tot_window = ext.params.PATCH_WINDOW_SECONDS + stft_par
        ons = np.arange(start=0, stop=stim.duration - tot_window, step=hop_size)
        return len(ons)

    audio_stim = AudioStim(join(AUDIO_DIR, 'crowd.mp3'))
    audio_filter = AudioResamplingFilter(target_sr=target_sr)
    audio_resampled = audio_filter.transform(audio_stim)

    # test with defaults and 44100 stimulus
    ext = AudiosetLabelExtractor(hop_size=hop_size)
    r_orig = ext.transform(audio_stim).to_df()
    assert r_orig.shape[0] == compute_expected_length(audio_stim, ext)
    assert r_orig.shape[1] == 525
    assert np.argmax(r_orig.to_numpy()[:,4:].mean(axis=0)) == 0
    assert r_orig['duration'][0] == .975
    assert all([np.isclose(r_orig['onset'][i] - r_orig['onset'][i-1], hop_size)
                for i in range(1,r_orig.shape[0])])

    # test resampled audio length and errors
    if target_sr >= 14500:
        r_resampled = ext.transform(audio_resampled).to_df()
        assert r_orig.shape[0] == r_resampled.shape[0]
    else:
        with pytest.raises(ValueError) as sr_error:
            ext.transform(audio_resampled)
        assert all([substr in str(sr_error.value) 
                    for substr in ['Upsample' , str(target_sr)]])

    # test top_n option
    ext_top_n = AudiosetLabelExtractor(top_n=top_n)
    r_top_n = ext_top_n.transform(audio_stim).to_df()
    assert r_top_n.shape[1] == ext_top_n.top_n + 4
    assert np.argmax(r_top_n.to_numpy()[:,4:].mean(axis=0)) == 0

    # test label subset
    labels = ['Speech', 'Silence', 'Harmonic', 'Bark', 'Music', 'Bell', 
              'Steam', 'Rain']
    ext_labels_only = AudiosetLabelExtractor(labels=labels)
    r_labels_only = ext_labels_only.transform(audio_stim).to_df()
    assert r_labels_only.shape[1] == len(labels) + 4
    
    # test top_n/labels error
    with pytest.raises(ValueError) as err:
        AudiosetLabelExtractor(top_n=10, labels=labels)
    assert 'Top_n and labels are mutually exclusive' in str(err.value)

def test_mfcc_energy_extractor():
    audio = AudioStim(join(AUDIO_DIR, 'barber.wav'))
    n_mels = 48

    ext = MFCCEnergyExtractor(register='low', n_mels=n_mels)
    data = ext.transform(audio).data.iloc[:, 4:]

    
    assert data.shape  == (1221, n_mels)
    assert np.isclose(data.iloc[0].sum(),695.7308991156441)
    
    ext = MFCCEnergyExtractor(register='high', n_mels=n_mels)
    data = ext.transform(audio).data.iloc[:, 4:]
    assert data.shape == (1221, n_mels)
    assert np.isclose(data.iloc[100].sum(),107.06728965998656)

    ext2 = MFCCEnergyExtractor(n_mfcc=64, n_coefs=8, hop_length=512, 
                               n_mels=n_mels, register='low')
    data = ext2.transform(audio).data.iloc[:, 4:]

    assert data.shape  == (1221, n_mels)
    assert np.isclose(data.iloc[650].sum(),884.4713338141073)

    ext2 = MFCCEnergyExtractor(n_mfcc=64, n_coefs=8, hop_length=512, 
                               n_mels=n_mels, register='high')
    data = ext2.transform(audio).data.iloc[:, 4:]
    assert data.shape == (1221, n_mels)
    assert np.isclose(data.iloc[601].sum(),data.iloc[601].sum())
