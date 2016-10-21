from os.path import join
import os
from .utils import get_test_data_path
from featurex.extractors.text import (DictionaryExtractor,
                                      PartOfSpeechExtractor,
                                      PredefinedDictionaryExtractor)
from featurex.extractors.audio import STFTExtractor, MeanAmplitudeExtractor
from featurex.extractors.image import (BrightnessExtractor,
                                        SharpnessExtractor,
                                        VibranceExtractor,
                                        SaliencyExtractor)
from featurex.extractors.video import DenseOpticalFlowExtractor
from featurex.extractors.api import (IndicoAPIExtractor,
                                        ClarifaiAPIExtractor)
from featurex.stimuli.text import ComplexTextStim
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
        stim.extract([td])


def test_text_extractor():
    stim = ComplexTextStim(join(TEXT_DIR, 'sample_text.txt'),
                           columns='to', default_duration=1)
    td = DictionaryExtractor(join(TEXT_DIR, 'test_lexical_dictionary.txt'),
                             variables=['length', 'frequency'])
    assert td.data.shape == (7, 2)
    timeline = stim.extract([td])
    df = timeline.to_df()
    assert np.isnan(df.iloc[0, 3])
    assert df.shape == (12, 4)
    target = df.query('name=="frequency" & onset==5')['value'].values
    assert target == 10.6


def test_predefined_dictionary_extractor():
    text = """enormous chunks of ice that have been frozen for thousands of
              years are breaking apart and melting away"""
    stim = ComplexTextStim.from_text(text)
    td = PredefinedDictionaryExtractor(['aoa/Freq_pm'])
    timeline = stim.extract([td])
    df = TimelineExporter.timeline_to_df(timeline)
    assert df.shape == (18, 4)


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
    timeline = stim.extract([ext])
    targets = [100., 150.]
    events = timeline.events
    values = [events[event].values[0].data["mean_amplitude"]
              for event in events.keys()]
    assert values == targets


def test_part_of_speech_extractor():
    stim = ComplexTextStim(join(TEXT_DIR, 'complex_stim_with_header.txt'))
    tl = stim.extract([PartOfSpeechExtractor()])
    df = tl.to_df()
    assert df.iloc[1, 3] == 'NN'
    assert df.shape == (4, 4)


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
    output = srt_stim.extract([ext])
    outdfKeys = set(output.to_df()['name'])
    outdfKeysCheck = set([
        'emotion_anger',
        'emotion_fear',
        'emotion_joy',
        'emotion_sadness',
        'emotion_surprise'])
    assert outdfKeys == outdfKeysCheck


@pytest.mark.skipif("'CLARIFAI_APP_ID' not in os.environ")
def test_clarifaiAPI_extractor():
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))
    ext = ClarifaiAPIExtractor()
    output = ext.transform(stim).data['tags']
    # Check success of request
    assert output['status_code'] == 'OK'
    # Check success of each image tagged
    for result in output['results']:
        assert result['status_code'] == 'OK'
        assert result['result']['tag']['classes']


def test_merge_extractor_results_by_features():
    np.random.seed(100)
    image_dir = join(get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))

    # Merge results for static Stims (no onsets)
    extractors = [BrightnessExtractor(), SharpnessExtractor(),
                  VibranceExtractor()]
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
