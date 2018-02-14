from os.path import join
from ..utils import get_test_data_path
from pliers.extractors import (LibrosaFeatureExtractor,
                               STFTAudioExtractor,
                               MeanAmplitudeExtractor,
                               SpectralCentroidExtractor,
                               SpectralBandwidthExtractor,
                               SpectralContrastExtractor,
                               SpectralRolloffExtractor,
                               PolyFeaturesExtractor,
                               RMSEExtractor,
                               ZeroCrossingRateExtractor,
                               ChromaSTFTExtractor,
                               ChromaCQTExtractor,
                               ChromaCENSExtractor,
                               MelspectrogramExtractor,
                               MFCCExtractor,
                               TonnetzExtractor,
                               TempogramExtractor)
from pliers.stimuli import (ComplexTextStim, AudioStim,
                            TranscribedAudioCompoundStim)
import numpy as np

AUDIO_DIR = join(get_test_data_path(), 'audio')


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
    ext = LibrosaFeatureExtractor(feature='rmse')
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 5)
    assert np.isclose(df['onset'][1], 0.04644)
    assert np.isclose(df['duration'][0], 0.04644)
    assert np.isclose(df['rmse'][0], 0.25663)


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


def test_rmse_extractor():
    audio = AudioStim(join(AUDIO_DIR, 'barber.wav'),
                      onset=1.0)
    ext = RMSEExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 5)
    assert np.isclose(df['onset'][1], 1.04644)
    assert np.isclose(df['duration'][0], 0.04644)
    assert np.isclose(df['rmse'][0], 0.25663)

    ext2 = RMSEExtractor(frame_length=1024, hop_length=256, center=False)
    df = ext2.transform(audio).to_df()
    assert df.shape == (2437, 5)
    assert np.isclose(df['onset'][1], 1.02322)
    assert np.isclose(df['duration'][0], 0.02322)
    assert np.isclose(df['rmse'][0], 0.25649)


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
    assert np.isclose(df['chroma_5'][0], 0.86870)

    ext = ChromaCQTExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 16)
    assert np.isclose(df['chroma_cqt_2'][0], 0.355324)

    ext = ChromaCENSExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (1221, 16)
    assert np.isclose(df['chroma_cens_2'][0], 0.137765)


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
    assert np.isclose(df['tonal_centroid_0'][0], -0.031784)


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
