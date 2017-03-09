from os.path import join
import os
from .utils import get_test_data_path, DummyExtractor
from pliers.extractors import (DictionaryExtractor, PartOfSpeechExtractor,
                               LengthExtractor, NumUniqueWordsExtractor,
                               PredefinedDictionaryExtractor, STFTAudioExtractor,
                               MeanAmplitudeExtractor, BrightnessExtractor,
                               SharpnessExtractor, VibranceExtractor,
                               SaliencyExtractor, DenseOpticalFlowExtractor,
                               IndicoAPITextExtractor, IndicoAPIImageExtractor,
                               ClarifaiAPIExtractor,
                               TensorFlowInceptionV3Extractor)
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
    stim = ImageStim(join(image_dir, 'button.jpg'))
    ext = LengthExtractor()
    result = ext.transform(stim).to_df()
    assert 'text_length' in result.columns
    assert result['text_length'][0] == 4


@pytest.mark.skipif("'WIT_AI_API_KEY' not in os.environ")
def test_implicit_stim_conversion2():
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'homer.wav'))
    ext = LengthExtractor()
    result = ext.transform(stim)
    first_word = result[0].to_df()
    assert 'text_length' in first_word.columns
    assert first_word['text_length'][0] > 0


@pytest.mark.skipif("'WIT_AI_API_KEY' not in os.environ")
def test_implicit_stim_conversion3():
    video_dir = join(get_test_data_path(), 'video')
    stim = VideoStim(join(video_dir, 'obama_speech.mp4'))
    ext = LengthExtractor()
    result = ext.transform(stim)
    first_word = result[0].to_df()
    # The word should be "today"
    assert 'text_length' in first_word.columns
    assert first_word['text_length'][0] == 5


def test_text_extractor():
    stim = ComplexTextStim(join(TEXT_DIR, 'sample_text.txt'),
                           columns='to', default_duration=1)
    td = DictionaryExtractor(join(TEXT_DIR, 'test_lexical_dictionary.txt'),
                             variables=['length', 'frequency'])
    assert td.data.shape == (7, 2)
    result = td.transform(stim)[2].to_df()
    assert np.isnan(result.iloc[0, 1])
    assert result.shape == (1, 4)
    assert np.isclose(result['frequency'][0], 11.729, 1e-5)


def test_text_length_extractor():
    stim = TextStim(text='hello world')
    ext = LengthExtractor()
    result = ext.transform(stim).to_df()
    assert 'text_length' in result.columns
    assert result['text_length'][0] == 11


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
    stim = AudioStim(join(audio_dir, 'barber.wav'))
    ext = STFTAudioExtractor(frame_size=1., spectrogram=False,
                             freq_bins=[(100, 300), (300, 3000), (3000, 20000)])
    result = ext.transform(stim)
    df = result.to_df()
    assert df.shape == (557, 5)


def test_mean_amplitude_extractor():
    audio = AudioStim(join(get_test_data_path(), 'audio', "barber_edited.wav"))
    text_file = join(get_test_data_path(), 'text', "wonderful_edited.srt")
    text = ComplexTextStim(text_file)
    stim = TranscribedAudioCompoundStim(audio=audio, text=text)
    ext = MeanAmplitudeExtractor()
    result = ext.transform(stim).to_df()
    targets = [-0.154661, 0.121521]
    assert np.allclose(result['mean_amplitude'], targets)


def test_part_of_speech_extractor():
    stim = ComplexTextStim(join(TEXT_DIR, 'complex_stim_with_header.txt'))
    result = PartOfSpeechExtractor().transform(stim).to_df()
    assert result.shape == (4, 6)
    assert 'NN' in result.columns
    assert result['NN'].sum() == 1
    assert result['VBD'][3] == 1


def test_brightness_extractor():
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))
    result = BrightnessExtractor().transform(stim).to_df()
    brightness = result['brightness'][0]
    assert np.isclose(brightness, 0.88784294, 1e-5)


def test_sharpness_extractor():
    pytest.importorskip('cv2')
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))
    result = SharpnessExtractor().transform(stim).to_df()
    sharpness = result['sharpness'][0]
    assert np.isclose(sharpness, 1.0, 1e-5)


def test_vibrance_extractor():
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))
    result = VibranceExtractor().transform(stim).to_df()
    color = result['vibrance'][0]
    assert np.isclose(color, 1370.65482988, 1e-5)


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
    stim = VideoStim(join(video_dir, 'small.mp4'))
    result = DenseOpticalFlowExtractor().transform(stim).to_df()
    target = result.query('onset==3.0')['total_flow']
    # Value returned by cv2 seems to change over versions, so use low precision
    assert np.isclose(target, 86248.05, 1e-4)


@pytest.mark.skipif("'INDICO_APP_KEY' not in os.environ")
def test_indico_api_text_extractor():

    ext = IndicoAPITextExtractor(api_key=os.environ['INDICO_APP_KEY'],
                             models=['emotion', 'personality'])

    # With ComplexTextStim input
    srtfile = join(get_test_data_path(), 'text', 'wonderful.srt')
    srt_stim = ComplexTextStim(srtfile)
    result = ext.transform(srt_stim).to_df()
    outdfKeysCheck = set([
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
        'personality_conscientiousness'])
    assert set(result.columns) == outdfKeysCheck

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
    result1 = ext.transform(stim1).to_df()

    outdfKeysCheck = set(['onset',
        'duration',
        'fer_Surprise',
        'fer_Neutral',
        'fer_Sad',
        'fer_Happy',
        'fer_Angry',
        'fer_Fear',
        'content_filtering'])

    assert set(result1.columns) == outdfKeysCheck
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


def test_merge_extractor_results_by_features():
    np.random.seed(100)
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))

    # Merge results for static Stims (no onsets)
    extractors = [BrightnessExtractor(), VibranceExtractor()]
    results = [e.transform(stim) for e in extractors]
    df = ExtractorResult.merge_features(results)

    de = DummyExtractor()
    de_names = ['Extractor1', 'Extractor2', 'Extractor3']
    results = [de.transform(stim, name) for name in de_names]
    df = ExtractorResult.merge_features(results)
    assert df.shape == (177, 14)
    assert df.columns.levels[1].unique().tolist() == ['duration', 0, 1, 2, '']
    cols = cols = ['onset', 'class', 'filename', 'history', 'stim']
    assert df.columns.levels[0].unique().tolist() == de_names + cols


def test_merge_extractor_results_by_stims():
    image_dir = join(get_test_data_path(), 'image')
    stim1 = ImageStim(join(image_dir, 'apple.jpg'))
    stim2 = ImageStim(join(image_dir, 'obama.jpg'))
    de = DummyExtractor()
    results = [de.transform(stim1), de.transform(stim2)]
    df = ExtractorResult.merge_stims(results)
    assert df.shape == (200, 6)
    assert set(df.columns.tolist()) == set(
        ['onset', 'duration', 0, 1, 2, 'stim'])
    assert set(df['stim'].unique()) == set(['obama.jpg', 'apple.jpg'])


def test_merge_extractor_results():
    np.random.seed(100)
    image_dir = join(get_test_data_path(), 'image')
    stim1 = ImageStim(join(image_dir, 'apple.jpg'))
    stim2 = ImageStim(join(image_dir, 'obama.jpg'))
    de = DummyExtractor()
    de_names = ['Extractor1', 'Extractor2', 'Extractor3']
    results = [de.transform(stim1, name) for name in de_names]
    results += [de.transform(stim2, name) for name in de_names]
    df = merge_results(results)
    assert df.shape == (355, 14)
    cols = ['onset', 'class', 'filename', 'history', 'stim']
    assert df.columns.levels[0].unique().tolist() == de_names + cols
    assert df.columns.levels[1].unique().tolist() == ['duration', 0, 1, 2, '']
    assert set(df['stim'].unique()) == set(['obama.jpg', 'apple.jpg'])


def test_tensor_flow_inception_v3_extractor():
    image_dir = join(get_test_data_path(), 'image')
    imgs = [join(image_dir, f) for f in ['apple.jpg', 'obama.jpg']]
    imgs = [ImageStim(im) for im in imgs]
    ext = TensorFlowInceptionV3Extractor()
    results = ext.transform(imgs)
    df = merge_results(results)
    assert len(df) == 2
    assert df.iloc[0][
        ('TensorFlowInceptionV3Extractor', 'label_1')] == 'Granny Smith'
    assert df.iloc[1][
        ('TensorFlowInceptionV3Extractor', 'score_2')] == '0.22610'
