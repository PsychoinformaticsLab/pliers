from os.path import join
from ..utils import get_test_data_path
from pliers.extractors import (STFTAudioExtractor,
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
from pliers.stimuli import (ComplexTextStim, AudioStim, TranscribedAudioCompoundStim)
import numpy as np

AUDIO_DIR = join(get_test_data_path(), 'audio')


def test_stft_extractor():
    stim = AudioStim(join(AUDIO_DIR, 'barber.wav'), onset=4.2)
    ext = STFTAudioExtractor(frame_size=1., spectrogram=False,
                             freq_bins=[(100, 300), (300, 3000), (3000, 20000)])
    result = ext.transform(stim)
    df = result.to_df()
    assert df.shape == (557, 5)
    assert df['onset'][0] == 4.2


def test_mean_amplitude_extractor():
    audio = AudioStim(join(AUDIO_DIR, "barber_edited.wav"))
    text_file = join(get_test_data_path(), 'text', "wonderful_edited.srt")
    text = ComplexTextStim(text_file)
    stim = TranscribedAudioCompoundStim(audio=audio, text=text)
    ext = MeanAmplitudeExtractor()
    result = ext.transform(stim).to_df()
    targets = [-0.154661, 0.121521]
    assert np.allclose(result['mean_amplitude'], targets)


def test_spectral_extractors():
    audio = AudioStim(join(AUDIO_DIR, "barber.wav"))
    ext = SpectralCentroidExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (4882, 3)
    assert np.isclose(df['onset'][1], 0.01161)
    assert np.isclose(df['duration'][0], 0.01161)
    assert np.isclose(df['spectral_centroid'][0], 817.53095)

    ext2 = SpectralCentroidExtractor(n_fft=1024, hop_length=256)
    df = ext2.transform(audio).to_df()
    assert df.shape == (9763, 3)
    assert np.isclose(df['onset'][1], 0.005805)
    assert np.isclose(df['duration'][0], 0.005805)
    assert np.isclose(df['spectral_centroid'][0], 1492.00515)

    ext = SpectralBandwidthExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (4882, 3)
    assert np.isclose(df['spectral_bandwidth'][0], 1056.66227)

    ext = SpectralContrastExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (4882, 9)
    assert np.isclose(df['spectral_contrast_band_4'][0], 25.09001)

    ext = SpectralRolloffExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (4882, 3)
    assert np.isclose(df['spectral_rolloff'][0], 1550.39063)


def test_polyfeatures_extractor():
    audio = AudioStim(join(AUDIO_DIR, "barber.wav"))
    ext = PolyFeaturesExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (4882, 4)
    assert np.isclose(df['onset'][1], 0.01161)
    assert np.isclose(df['duration'][0], 0.01161)
    assert np.isclose(df['coefficient_0'][0], -7.795e-5)

    ext2 = PolyFeaturesExtractor(order=3)
    df = ext2.transform(audio).to_df()
    assert df.shape == (4882, 6)
    assert np.isclose(df['coefficient_3'][2], 20.77778)


def test_rmse_extractor():
    audio = AudioStim(join(AUDIO_DIR, "barber.wav"),
                      onset=1.0)
    ext = RMSEExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (4882, 3)
    assert np.isclose(df['onset'][1], 1.01161)
    assert np.isclose(df['duration'][0], 0.01161)
    assert np.isclose(df['rmse'][0], 0.226572)

    ext2 = RMSEExtractor(frame_length=1024, hop_length=256, center=False)
    df = ext2.transform(audio).to_df()
    assert df.shape == (9759, 3)
    assert np.isclose(df['onset'][1], 1.005805)
    assert np.isclose(df['duration'][0], 0.005805)
    assert np.isclose(df['rmse'][0], 0.22648)


def test_zcr_extractor():
    audio = AudioStim(join(AUDIO_DIR, "barber.wav"),
                      onset=2.0)
    ext = ZeroCrossingRateExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (4882, 3)
    assert np.isclose(df['onset'][1], 2.01161)
    assert np.isclose(df['duration'][0], 0.01161)
    assert np.isclose(df['zero_crossing_rate'][0], 0.0234375)

    ext2 = ZeroCrossingRateExtractor(frame_length=1024, hop_length=256,
                                     center=False, pad=True)
    df = ext2.transform(audio).to_df()
    assert df.shape == (9759, 3)
    assert np.isclose(df['onset'][1], 2.005805)
    assert np.isclose(df['duration'][0], 0.005805)
    assert np.isclose(df['zero_crossing_rate'][0], 0.047852)


def test_chroma_extractors():
    audio = AudioStim(join(AUDIO_DIR, "barber.wav"))
    ext = ChromaSTFTExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (4882, 14)
    assert np.isclose(df['onset'][1], 0.01161)
    assert np.isclose(df['duration'][0], 0.01161)
    assert np.isclose(df['chroma_2'][0], 0.417595)

    ext2 = ChromaSTFTExtractor(n_chroma=6, n_fft=1024, hop_length=256)
    df = ext2.transform(audio).to_df()
    assert df.shape == (9763, 8)
    assert np.isclose(df['onset'][1], 0.005805)
    assert np.isclose(df['duration'][0], 0.005805)
    assert np.isclose(df['chroma_5'][0], 0.732480)

    ext = ChromaCQTExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (4882, 14)
    assert np.isclose(df['chroma_cqt_2'][0], 0.286443)

    ext = ChromaCENSExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (4882, 14)
    assert np.isclose(df['chroma_cens_2'][0], 0.217814)


def test_melspectrogram_extractor():
    audio = AudioStim(join(AUDIO_DIR, "barber.wav"))
    ext = MelspectrogramExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (4882, 130)
    assert np.isclose(df['onset'][1], 0.01161)
    assert np.isclose(df['duration'][0], 0.01161)
    assert np.isclose(df['mel_3'][0], 0.553125)

    ext2 = MelspectrogramExtractor(n_mels=15)
    df = ext2.transform(audio).to_df()
    assert df.shape == (4882, 17)
    assert np.isclose(df['mel_4'][2], 3.24429)


def test_mfcc_extractor():
    audio = AudioStim(join(AUDIO_DIR, "barber.wav"))
    ext = MFCCExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (4882, 22)
    assert np.isclose(df['onset'][1], 0.01161)
    assert np.isclose(df['duration'][0], 0.01161)
    assert np.isclose(df['mfcc_3'][0], 5.98247)

    ext2 = MFCCExtractor(n_mfcc=15)
    df = ext2.transform(audio).to_df()
    assert df.shape == (4882, 17)
    assert np.isclose(df['mfcc_14'][2], -7.41533)


def test_tonnetz_extractor():
    audio = AudioStim(join(AUDIO_DIR, "barber.wav"))
    ext = TonnetzExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (4882, 8)
    assert np.isclose(df['onset'][1], 0.01161)
    assert np.isclose(df['duration'][0], 0.01161)
    assert np.isclose(df['tonal_centroid_0'][0], -0.0264436)


def test_tempogram_extractor():
    audio = AudioStim(join(AUDIO_DIR, "barber.wav"))
    ext = TempogramExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (4882, 386)
    assert np.isclose(df['onset'][1], 0.01161)
    assert np.isclose(df['duration'][0], 0.01161)
    assert np.isclose(df['tempo_1'][0], 0.773760)

    ext2 = TempogramExtractor(win_length=300)
    df = ext2.transform(audio).to_df()
    assert df.shape == (4882, 302)
    assert np.isclose(df['tempo_1'][2], 0.756967)
