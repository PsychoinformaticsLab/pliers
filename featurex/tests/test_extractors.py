from os.path import join
import os
from .utils import get_test_data_path
from featurex.extractors.text import (DictionaryExtractor,
                                      PartOfSpeechExtractor,
                                      LengthExtractor,
                                      NumUniqueWordsExtractor,
                                      PredefinedDictionaryExtractor)
from featurex.extractors.audio import STFTExtractor, MeanAmplitudeExtractor
from featurex.extractors.image import (BrightnessExtractor,
                                        SharpnessExtractor,
                                        VibranceExtractor,
                                        SaliencyExtractor)
from featurex.extractors.video import DenseOpticalFlowExtractor
from featurex.extractors.api import (IndicoAPIExtractor,
                                        ClarifaiAPIExtractor)
from featurex.stimuli.text import TextStim, ComplexTextStim
from featurex.stimuli.video import ImageStim, VideoStim
from featurex.stimuli.audio import AudioStim, TranscribedAudioStim
from featurex.export import TimelineExporter
from featurex.support.download import download_nltk_data
from featurex.extractors import Extractor, ExtractorResult, merge_results
import numpy as np
import pytest
from copy import deepcopy

TEXT_DIR = join(get_test_data_path(), 'text')


class DummyExtractor(Extractor):
    target = ImageStim

    def _extract(self, stim, name=None, n_rows=100, n_cols=3, max_time=1000):
        data = np.random.randint(0, 1000, (n_rows, n_cols))
        onsets = np.random.choice(n_rows*2, n_rows, False)
        if name is not None:
            self.name = name
        return ExtractorResult(data, stim, deepcopy(self), onsets=onsets)


@pytest.fixture(scope='module')
def get_nltk():
    download_nltk_data()


def test_check_target_type():
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'barber.wav'))
    td = DictionaryExtractor(join(TEXT_DIR, 'test_lexical_dictionary.txt'),
                             variables=['length', 'frequency'])
    with pytest.raises(TypeError):
        td.transform(stim)


def test_text_length_extractor():
    stim = TextStim(text='hello world')
    ext = LengthExtractor()
    result = ext.extract(stim).to_df()
    assert 'text_length' in result.columns
    assert result['text_length'][0] == 11


def test_unique_words_extractor():
    stim = TextStim(text='hello hello world')
    ext = NumUniqueWordsExtractor()
    result = ext.extract(stim).to_df()
    print result
    assert 'num_unique_words' in result.columns
    assert result['num_unique_words'][0] == 2


def test_dictionary_extractor():
    td = DictionaryExtractor(join(TEXT_DIR, 'test_lexical_dictionary.txt'),
                             variables=['length', 'frequency'])
    assert td.data.shape == (7, 2)

    stim = TextStim(text='annotation')
    result = td.extract(stim).to_df()
    assert np.isnan(result['onset'][0])
    assert 'length' in result.columns
    assert result['length'][0] == 10

    stim2 = TextStim(text='some')
    result = td.extract(stim2).to_df()
    assert np.isnan(result['onset'][0])
    assert 'frequency' in result.columns
    assert np.isnan(result['frequency'][0])


def test_predefined_dictionary_extractor():
    stim = TextStim(text='enormous')
    td = PredefinedDictionaryExtractor(['aoa/Freq_pm'])
    result = td.extract(stim).to_df()
    assert result.shape == (1, 3)
    assert 'aoa_Freq_pm' in result.columns
    assert np.isclose(result['aoa_Freq_pm'][0], 10.313725)


def test_stft_extractor():
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'barber.wav'))
    ext = STFTExtractor(frame_size=1., spectrogram=False,
                        bins=[(100, 300), (300, 3000), (3000, 20000)])
    result = ext.extract(stim)
    df = result.to_df()
    assert df.shape == (557, 5)


def test_mean_amplitude_extractor():
    audio_dir = join(get_test_data_path(), 'audio')
    text_dir = join(get_test_data_path(), 'text')
    stim = TranscribedAudioStim(join(audio_dir, "barber_edited.wav"),
                                join(text_dir, "wonderful_edited.srt"))
    ext = MeanAmplitudeExtractor()
    result = ext.extract(stim).to_df()
    targets = [100., 150.]
    assert np.array_equal(result['mean_amplitude'], targets)


def test_part_of_speech_extractor():
    stim = ComplexTextStim(join(TEXT_DIR, 'complex_stim_with_header.txt'))
    result = PartOfSpeechExtractor().extract(stim).to_df()
    assert result.shape == (4, 6)
    assert 'NN' in result.columns
    assert result['NN'].sum() == 1
    assert result['VBD'][3] == 1


def test_brightness_extractor():
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))
    result = BrightnessExtractor().extract(stim).to_df()
    brightness = result['brightness'][0]
    assert np.isclose(brightness, 0.88784294)


def test_sharpness_extractor():
    pytest.importorskip('cv2')
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))
    result = SharpnessExtractor().extract(stim).to_df()
    sharpness = result['sharpness'][0]
    assert np.isclose(sharpness, 1.0)


def test_vibrance_extractor():
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))
    result = VibranceExtractor().extract(stim).to_df()
    color = result['vibrance'][0]
    assert np.isclose(color, 1370.65482988)


def test_saliency_extractor():
    pytest.importorskip('cv2')
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))
    result = SaliencyExtractor().extract(stim).to_df()
    ms = result['max_saliency'][0]
    assert np.isclose(ms, 0.99669953)
    sf = result['frac_high_saliency'][0]
    assert np.isclose(sf, 0.27461971)


def test_optical_flow_extractor():
    pytest.importorskip('cv2')
    video_dir = join(get_test_data_path(), 'video')
    stim = VideoStim(join(video_dir, 'small.mp4'))
    result = DenseOpticalFlowExtractor().extract(stim).to_df()
    target = result.query('onset==3.0')['total_flow']
    assert np.isclose(target, 86248.05)


@pytest.mark.skipif("'INDICO_APP_KEY' not in os.environ")
def test_indicoAPI_extractor():
    srtfile = join(get_test_data_path(), 'text', 'wonderful.srt')
    srt_stim = ComplexTextStim(srtfile)
    ext = IndicoAPIExtractor(
        api_key=os.environ['INDICO_APP_KEY'], model='emotion')
    result = ext.extract(srt_stim).to_df()
    outdfKeysCheck = set([
        'onset',
        'duration',
        'emotion_anger',
        'emotion_fear',
        'emotion_joy',
        'emotion_sadness',
        'emotion_surprise'])
    assert set(result.columns) == outdfKeysCheck


@pytest.mark.skipif("'CLARIFAI_APP_ID' not in os.environ")
def test_clarifaiAPI_extractor():
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))
    result = ClarifaiAPIExtractor().extract(stim).to_df()
    assert result['apple'][0] > 0.5
    assert result.ix[:,5][0] > 0.0


def test_merge_extractor_results_by_features():
    np.random.seed(100)
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))

    # Merge results for static Stims (no onsets)
    extractors = [BrightnessExtractor(), VibranceExtractor()]
    results = [e.extract(stim) for e in extractors]
    df = ExtractorResult.merge_features(results)

    de = DummyExtractor()
    de_names = ['Extractor1', 'Extractor2', 'Extractor3']
    results = [de.extract(stim, name) for name in de_names]
    df = ExtractorResult.merge_features(results)
    assert df.shape == (177, 10)
    assert df.columns.levels[1].unique().tolist() == ['duration', 0, 1, 2, '']
    assert df.columns.levels[0].unique().tolist() == de_names + ['onset', 'stim']


def test_merge_extractor_results_by_stims():
    image_dir = join(get_test_data_path(), 'image')
    stim1 = ImageStim(join(image_dir, 'apple.jpg'))
    stim2 = ImageStim(join(image_dir, 'obama.jpg'))
    de = DummyExtractor()
    results = [de.extract(stim1), de.extract(stim2)]
    df = ExtractorResult.merge_stims(results)
    assert df.shape == (200, 5)
    assert df.columns.tolist() == ['onset', 'duration', 0, 1, 2]
    assert set(df.index.levels[1].unique()) == set(['obama.jpg', 'apple.jpg'])


def test_merge_extractor_results():
    np.random.seed(100)
    image_dir = join(get_test_data_path(), 'image')
    stim1 = ImageStim(join(image_dir, 'apple.jpg'))
    stim2 = ImageStim(join(image_dir, 'obama.jpg'))
    de = DummyExtractor()
    de_names = ['Extractor1', 'Extractor2', 'Extractor3']
    results = [de.extract(stim1, name) for name in de_names]
    results += [de.extract(stim2, name) for name in de_names]
    df = merge_results(results)
    assert df.shape == (355, 10)
    assert df.columns.levels[0].unique().tolist() == de_names + ['onset', 'stim']
    assert df.columns.levels[1].unique().tolist() == ['duration', 0, 1, 2, '']
    assert set(df.index.levels[1].unique()) == set(['obama.jpg', 'apple.jpg'])
