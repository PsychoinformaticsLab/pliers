from os.path import join
import os
from .utils import get_test_data_path, DummyExtractor
from pliers.extractors import (DictionaryExtractor, PartOfSpeechExtractor,
                               LengthExtractor, NumUniqueWordsExtractor,
                               PredefinedDictionaryExtractor,
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
                               TempogramExtractor,
                               BrightnessExtractor,
                               SharpnessExtractor, VibranceExtractor,
                               SaliencyExtractor, FarnebackOpticalFlowExtractor,
                               IndicoAPITextExtractor, IndicoAPIImageExtractor,
                               ClarifaiAPIExtractor,
                               TensorFlowInceptionV3Extractor,
                               TextVectorizerExtractor,
                               WordEmbeddingExtractor,
                               VADERSentimentExtractor)
from pliers.stimuli import (TextStim, ComplexTextStim, ImageStim, VideoStim,
                            AudioStim, TranscribedAudioCompoundStim)
from pliers.support.download import download_nltk_data
from pliers.extractors.base import ExtractorResult, merge_results
import numpy as np
import pytest

TEXT_DIR = join(get_test_data_path(), 'text')


@pytest.fixture(scope='module')
def get_nltk():
    download_nltk_data()


def test_check_target_type():
    stim = ComplexTextStim(join(TEXT_DIR, 'sample_text.txt'),
                           columns='to', default_duration=1)
    td = SharpnessExtractor()
    with pytest.raises(TypeError):
        td.transform(stim)


def test_implicit_stim_iteration():
    np.random.seed(100)
    image_dir = join(get_test_data_path(), 'image')
    stim1 = ImageStim(join(image_dir, 'apple.jpg'))
    stim2 = ImageStim(join(image_dir, 'obama.jpg'))
    de = DummyExtractor()
    results = de.transform([stim1, stim2])
    assert len(results) == 2
    assert isinstance(results[0], ExtractorResult)


def test_implicit_stim_conversion():
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'button.jpg'), onset=4.2)
    ext = LengthExtractor()
    result = ext.transform(stim).to_df()
    assert 'text_length' in result.columns
    assert result['text_length'][0] == 4
    assert result['onset'][0] == 4.2


@pytest.mark.skipif("'WIT_AI_API_KEY' not in os.environ")
def test_implicit_stim_conversion2():
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'homer.wav'), onset=4.2)
    ext = LengthExtractor()
    result = ext.transform(stim)
    first_word = result[0].to_df()
    assert 'text_length' in first_word.columns
    assert first_word['text_length'][0] > 0
    assert first_word['onset'][0] >= 4.2


@pytest.mark.skipif("'WIT_AI_API_KEY' not in os.environ")
def test_implicit_stim_conversion3():
    video_dir = join(get_test_data_path(), 'video')
    stim = VideoStim(join(video_dir, 'obama_speech.mp4'), onset=4.2)
    ext = LengthExtractor()
    result = ext.transform(stim)
    first_word = result[0].to_df()
    # The word should be "today"
    assert 'text_length' in first_word.columns
    assert first_word['text_length'][0] == 5
    assert first_word['onset'][0] >= 4.2


def test_text_extractor():
    stim = ComplexTextStim(join(TEXT_DIR, 'sample_text.txt'),
                           columns='to', default_duration=1)
    td = DictionaryExtractor(join(TEXT_DIR, 'test_lexical_dictionary.txt'),
                             variables=['length', 'frequency'])
    assert td.data.shape == (7, 2)
    result = td.transform(stim)[2].to_df()
    assert result.iloc[0, 1] == 1
    assert result.shape == (1, 4)
    assert np.isclose(result['frequency'][0], 11.729, 1e-5)


def test_text_length_extractor():
    stim = TextStim(text='hello world', onset=4.2, duration=1)
    ext = LengthExtractor()
    result = ext.transform(stim).to_df()
    assert 'text_length' in result.columns
    assert result['text_length'][0] == 11
    assert result['onset'][0] == 4.2
    assert result['duration'][0] == 1


def test_unique_words_extractor():
    stim = TextStim(text='hello hello world')
    ext = NumUniqueWordsExtractor()
    result = ext.transform(stim).to_df()
    assert 'num_unique_words' in result.columns
    assert result['num_unique_words'][0] == 2


def test_dictionary_extractor():
    td = DictionaryExtractor(join(TEXT_DIR, 'test_lexical_dictionary.txt'),
                             variables=['length', 'frequency'])
    assert td.data.shape == (7, 2)

    stim = TextStim(text='annotation')
    result = td.transform(stim).to_df()
    assert np.isnan(result['onset'][0])
    assert 'length' in result.columns
    assert result['length'][0] == 10

    stim2 = TextStim(text='some')
    result = td.transform(stim2).to_df()
    assert np.isnan(result['onset'][0])
    assert 'frequency' in result.columns
    assert np.isnan(result['frequency'][0])


def test_predefined_dictionary_extractor():
    stim = TextStim(text='enormous')
    td = PredefinedDictionaryExtractor(['aoa/Freq_pm'])
    result = td.transform(stim).to_df()
    assert result.shape == (1, 3)
    assert 'aoa_Freq_pm' in result.columns
    assert np.isclose(result['aoa_Freq_pm'][0], 10.313725, 1e-5)


def test_stft_extractor():
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'barber.wav'), onset=4.2)
    ext = STFTAudioExtractor(frame_size=1., spectrogram=False,
                             freq_bins=[(100, 300), (300, 3000), (3000, 20000)])
    result = ext.transform(stim)
    df = result.to_df()
    assert df.shape == (557, 5)
    assert df['onset'][0] == 4.2


def test_mean_amplitude_extractor():
    audio = AudioStim(join(get_test_data_path(), 'audio', "barber_edited.wav"))
    text_file = join(get_test_data_path(), 'text', "wonderful_edited.srt")
    text = ComplexTextStim(text_file)
    stim = TranscribedAudioCompoundStim(audio=audio, text=text)
    ext = MeanAmplitudeExtractor()
    result = ext.transform(stim).to_df()
    targets = [-0.154661, 0.121521]
    assert np.allclose(result['mean_amplitude'], targets)


def test_spectral_extractors():
    audio = AudioStim(join(get_test_data_path(), 'audio', "barber.wav"))
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
    audio = AudioStim(join(get_test_data_path(), 'audio', "barber.wav"))
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
    audio = AudioStim(join(get_test_data_path(), 'audio', "barber.wav"),
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
    audio = AudioStim(join(get_test_data_path(), 'audio', "barber.wav"),
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
    audio = AudioStim(join(get_test_data_path(), 'audio', "barber.wav"))
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
    audio = AudioStim(join(get_test_data_path(), 'audio', "barber.wav"))
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
    audio = AudioStim(join(get_test_data_path(), 'audio', "barber.wav"))
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
    audio = AudioStim(join(get_test_data_path(), 'audio', "barber.wav"))
    ext = TonnetzExtractor()
    df = ext.transform(audio).to_df()
    assert df.shape == (4882, 8)
    assert np.isclose(df['onset'][1], 0.01161)
    assert np.isclose(df['duration'][0], 0.01161)
    assert np.isclose(df['tonal_centroid_0'][0], -0.0264436)


def test_tempogram_extractor():
    audio = AudioStim(join(get_test_data_path(), 'audio', "barber.wav"))
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


def test_part_of_speech_extractor():
    import nltk
    nltk.download('tagsets')
    stim = ComplexTextStim(join(TEXT_DIR, 'complex_stim_with_header.txt'))
    result = merge_results(PartOfSpeechExtractor().transform(stim), extractor_names=False)
    assert result.shape == (4, 52)
    assert result['NN'].sum() == 1
    assert result['VBD'][3] == 1


def test_word_embedding_extractor():
    pytest.importorskip('gensim')
    stims = [TextStim(text='this'), TextStim(text='sentence')]
    ext = WordEmbeddingExtractor(join(TEXT_DIR, 'simple_vectors.bin'),
                                 binary=True)
    result = merge_results(ext.transform(stims))
    assert ('WordEmbeddingExtractor', 'embedding_dim99') in result.columns
    assert 0.001091 in result[('WordEmbeddingExtractor', 'embedding_dim0')]


def test_vectorizer_extractor():
    pytest.importorskip('sklearn')
    stim = TextStim(join(TEXT_DIR, 'scandal.txt'))
    result = TextVectorizerExtractor().transform(stim).to_df()
    assert 'woman' in result.columns
    assert result['woman'][0] == 3

    from sklearn.feature_extraction.text import TfidfVectorizer
    custom_vectorizer = TfidfVectorizer()
    ext = TextVectorizerExtractor(vectorizer=custom_vectorizer)
    stim2 = TextStim(join(TEXT_DIR, 'simple_text.txt'))
    result = merge_results(ext.transform([stim, stim2]))
    assert ('TextVectorizerExtractor', 'woman') in result.columns
    assert 0.129568189476 in result[('TextVectorizerExtractor', 'woman')]

    ext = TextVectorizerExtractor(vectorizer='CountVectorizer',
                                  analyzer='char_wb',
                                  ngram_range=(2, 2))
    result = ext.transform(stim).to_df()
    assert 'wo' in result.columns
    assert result['wo'][0] == 6


def test_vader_sentiment_extractor():
    stim = TextStim(join(TEXT_DIR, 'scandal.txt'))
    ext = VADERSentimentExtractor()
    result = ext.transform(stim).to_df()
    assert result['sentiment_neu'][0] == 0.752

    stim2 = TextStim(text='VADER is smart, handsome, and funny!')
    result2 = ext.transform(stim2).to_df()
    assert result2['sentiment_pos'][0] == 0.752
    assert result2['sentiment_neg'][0] == 0.0
    assert result2['sentiment_neu'][0] == 0.248
    assert result2['sentiment_compound'][0] == 0.8439


def test_brightness_extractor():
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'), onset=4.2, duration=1)
    result = BrightnessExtractor().transform(stim).to_df()
    brightness = result['brightness'][0]
    assert np.isclose(brightness, 0.88784294, 1e-5)
    assert result['onset'][0] == 4.2
    assert result['duration'][0] == 1


def test_sharpness_extractor():
    pytest.importorskip('cv2')
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'), onset=4.2, duration=1)
    result = SharpnessExtractor().transform(stim).to_df()
    sharpness = result['sharpness'][0]
    assert np.isclose(sharpness, 1.0, 1e-5)
    assert result['onset'][0] == 4.2
    assert result['duration'][0] == 1


def test_vibrance_extractor():
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'), onset=4.2, duration=1)
    result = VibranceExtractor().transform(stim).to_df()
    color = result['vibrance'][0]
    assert np.isclose(color, 1370.65482988, 1e-5)
    assert result['onset'][0] == 4.2
    assert result['duration'][0] == 1


def test_saliency_extractor():
    pytest.importorskip('cv2')
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))
    result = SaliencyExtractor().transform(stim).to_df()
    ms = result['max_saliency'][0]
    assert np.isclose(ms, 0.99669953, 1e-5)
    sf = result['frac_high_saliency'][0]
    assert np.isclose(sf, 0.27461971, 1e-5)


def test_optical_flow_extractor():
    pytest.importorskip('cv2')
    video_dir = join(get_test_data_path(), 'video')
    stim = VideoStim(join(video_dir, 'small.mp4'), onset=4.2)
    result = FarnebackOpticalFlowExtractor().transform(stim).to_df()
    target = result.query('onset==7.2')['total_flow']
    # Value returned by cv2 seems to change over versions, so use low precision
    assert np.isclose(target, 86248.05, 1e-4)


@pytest.mark.skipif("'INDICO_APP_KEY' not in os.environ")
def test_indico_api_text_extractor():

    ext = IndicoAPITextExtractor(api_key=os.environ['INDICO_APP_KEY'],
                                 models=['emotion', 'personality'])

    # With ComplexTextStim input
    srtfile = join(get_test_data_path(), 'text', 'wonderful.srt')
    srt_stim = ComplexTextStim(srtfile, onset=4.2)
    result = ExtractorResult.merge_stims(ext.transform(srt_stim))
    outdfKeysCheck = {
        'onset',
        'duration',
        'emotion_anger',
        'emotion_fear',
        'emotion_joy',
        'emotion_sadness',
        'emotion_surprise',
        'personality_openness',
        'personality_extraversion',
        'personality_agreeableness',
        'personality_conscientiousness'}
    meta_columns = {'source_file',
                    'history',
                    'class',
                    'filename'}
    assert set(result.columns) - set(['stim_name']) == outdfKeysCheck | meta_columns
    assert result['onset'][1] == 92.622

    # With TextStim input
    ts = TextStim(text="It's a wonderful life.")
    result = ext.transform(ts).to_df()
    assert set(result.columns) == outdfKeysCheck
    assert len(result) == 1


@pytest.mark.skipif("'INDICO_APP_KEY' not in os.environ")
def test_indico_api_image_extractor():

    ext = IndicoAPIImageExtractor(api_key=os.environ['INDICO_APP_KEY'],
                                  models=['fer', 'content_filtering'])

    image_dir = join(get_test_data_path(), 'image')
    stim1 = ImageStim(join(image_dir, 'apple.jpg'))
    result1 = ExtractorResult.merge_stims(ext.transform([stim1, stim1]))

    outdfKeysCheck = {
        'onset',
        'duration',
        'fer_Surprise',
        'fer_Neutral',
        'fer_Sad',
        'fer_Happy',
        'fer_Angry',
        'fer_Fear',
        'content_filtering'}
    meta_columns = {'source_file',
                    'history',
                    'class',
                    'filename'}

    assert set(result1.columns) - set(['stim_name']) == outdfKeysCheck | meta_columns
    assert result1['content_filtering'][0] < 0.1

    stim2 = ImageStim(join(image_dir, 'obama.jpg'))
    result2 = ext.transform(stim2).to_df()
    assert set(result2.columns) == outdfKeysCheck
    assert result2['fer_Happy'][0] > 0.7


@pytest.mark.skipif("'CLARIFAI_APP_ID' not in os.environ")
def test_clarifai_api_extractor():
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))
    result = ClarifaiAPIExtractor().transform(stim).to_df()
    assert result['apple'][0] > 0.5
    assert result.ix[:, 5][0] > 0.0

    result = ClarifaiAPIExtractor(max_concepts=5).transform(stim).to_df()
    assert result.shape == (1, 7)

    result = ClarifaiAPIExtractor(min_value=0.9).transform(stim).to_df()
    assert all(np.isnan(d) or d > 0.9 for d in result.values[0])

    concepts = ['cat', 'dog']
    result = ClarifaiAPIExtractor(select_concepts=concepts).transform(stim)
    result = result.to_df()
    assert result.shape == (1, 4)
    assert 'cat' in result.columns and 'dog' in result.columns


@pytest.mark.skipif("'CLARIFAI_APP_ID' not in os.environ")
def test_clarifai_api_extractor_batch():
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))
    stim2 = ImageStim(join(image_dir, 'obama.jpg'))
    ext = ClarifaiAPIExtractor()
    results = ext.transform([stim, stim2])
    results = merge_results(results)
    assert results[('ClarifaiAPIExtractor', 'apple')][0] > 0.5 or \
        results[('ClarifaiAPIExtractor', 'apple')][1] > 0.5

    # This takes too long to execute
    # video = VideoStim(join(get_test_data_path(), 'video', 'small.mp4'))
    # results = ExtractorResult.merge_stims(ext.transform(video))
    # assert 'Lego' in results.columns and 'robot' in results.columns


def test_merge_extractor_results_by_features():
    np.random.seed(100)
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))

    # Merge results for static Stims (no onsets)
    extractors = [BrightnessExtractor(), VibranceExtractor()]
    results = [e.transform(stim) for e in extractors]
    df = ExtractorResult.merge_features(results)

    de_names = ['Extractor1', 'Extractor2', 'Extractor3']
    des = [DummyExtractor(name=name) for name in de_names]
    results = [de.transform(stim) for de in des]
    df = ExtractorResult.merge_features(results)
    assert df.shape == (177, 16)
    assert df.columns.levels[1].unique().tolist() == [0, 1, 2, '']
    cols = ['onset', 'duration', 'class', 'filename', 'history', 'stim_name',
            'source_file']
    assert df.columns.levels[0].unique().tolist() == de_names + cols


def test_merge_extractor_results_by_stims():
    image_dir = join(get_test_data_path(), 'image')
    stim1 = ImageStim(join(image_dir, 'apple.jpg'))
    stim2 = ImageStim(join(image_dir, 'obama.jpg'))
    de = DummyExtractor()
    results = [de.transform(stim1), de.transform(stim2)]
    df = ExtractorResult.merge_stims(results)
    assert df.shape == (200, 10)
    assert set(df.columns.tolist()) == set(
        ['onset', 'duration', 0, 1, 2, 'stim_name', 'class', 'filename',
         'history', 'source_file'])
    assert set(df['stim_name'].unique()) == set(['obama.jpg', 'apple.jpg'])


def test_merge_extractor_results():
    np.random.seed(100)
    image_dir = join(get_test_data_path(), 'image')
    stim1 = ImageStim(join(image_dir, 'apple.jpg'))
    stim2 = ImageStim(join(image_dir, 'obama.jpg'))
    de_names = ['Extractor1', 'Extractor2', 'Extractor3']
    des = [DummyExtractor(name=name) for name in de_names]
    results = [de.transform(stim1) for de in des]
    results += [de.transform(stim2) for de in des]
    df = merge_results(results)
    assert df.shape == (354, 16)
    cols = ['onset', 'duration', 'class', 'filename', 'history', 'stim_name',
            'source_file']
    assert df.columns.levels[0].unique().tolist() == de_names + cols
    assert df.columns.levels[1].unique().tolist() == [0, 1, 2, '']
    assert set(df['stim_name'].unique()) == set(['obama.jpg', 'apple.jpg'])


def test_merge_extractor_results_flattened():
    np.random.seed(100)
    image_dir = join(get_test_data_path(), 'image')
    stim1 = ImageStim(join(image_dir, 'apple.jpg'))
    stim2 = ImageStim(join(image_dir, 'obama.jpg'))
    de_names = ['Extractor1', 'Extractor2', 'Extractor3']
    des = [DummyExtractor(name=name) for name in de_names]
    results = [de.transform(stim1) for de in des]
    results += [de.transform(stim2) for de in des]
    df = merge_results(results, flatten_columns=True)
    de_cols = ['Extractor1_0', 'Extractor1_1', 'Extractor1_2',
               'Extractor2_0', 'Extractor2_1', 'Extractor2_2',
               'Extractor3_0', 'Extractor3_1', 'Extractor3_2']
    assert df.shape == (354, 16)
    cols = ['onset', 'class', 'filename', 'history', 'stim_name', 'duration',
            'source_file']
    assert set(df.columns.unique().tolist()) == set(cols + de_cols)


def test_tensor_flow_inception_v3_extractor():
    image_dir = join(get_test_data_path(), 'image')
    imgs = [join(image_dir, f) for f in ['apple.jpg', 'obama.jpg']]
    imgs = [ImageStim(im, onset=4.2, duration=1) for im in imgs]
    ext = TensorFlowInceptionV3Extractor()
    results = ext.transform(imgs)
    df = merge_results(results)
    assert len(df) == 2
    assert ('TensorFlowInceptionV3Extractor', 'Granny Smith') in df.columns
    assert 0.22610 in df[
        ('TensorFlowInceptionV3Extractor', 'Windsor tie')].values
    assert 4.2 in df[('onset', '')].values
    assert 1 in df[('duration', '')].values
