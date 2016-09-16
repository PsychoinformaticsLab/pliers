from os.path import join
import os
from .utils import _get_test_data_path
from featurex.extractors.text import (DictionaryExtractor,
                                      PartOfSpeechExtractor,
                                      PredefinedDictionaryExtractor)
from featurex.extractors.audio import STFTExtractor, MeanAmplitudeExtractor
from featurex.extractors.image import TesseractExtractor
from featurex.extractors.image import SaliencyExtractor
from featurex.extractors.api import ClarifaiAPIExtractor
from featurex.extractors.api import IndicoAPIExtractor
from featurex.stimuli.text import ComplexTextStim
from featurex.stimuli.video import ImageStim
from featurex.stimuli.audio import AudioStim, TranscribedAudioStim
from featurex.export import TimelineExporter
from featurex.extractors import get_extractor
from featurex.support.download import download_nltk_data
import numpy as np
import pytest

TEXT_DIR = join(_get_test_data_path(), 'text')

@pytest.fixture(scope='module')
def get_nltk():
    download_nltk_data()

def test_check_target_type():
    audio_dir = join(_get_test_data_path(), 'audio')
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
    audio_dir = join(_get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'barber.wav'))
    ext = STFTExtractor(frame_size=1., spectrogram=False,
                        bins=[(100, 300), (300, 3000), (3000, 20000)])
    timeline = stim.extract([ext])
    df = timeline.to_df('long')
    assert df.shape == (1671, 4)

def test_mean_amplitude_extractor():
    audio_dir = join(_get_test_data_path(), 'audio')
    text_dir = join(_get_test_data_path(), 'text')
    stim = TranscribedAudioStim(join(audio_dir, "barber_edited.wav"),
                                join(text_dir, "wonderful_edited.srt"))
    ext = MeanAmplitudeExtractor()
    timeline = stim.extract([ext])
    targets = [100., 150.]
    events = timeline.events
    values = [events[event].values[0].data["mean_amplitude"] for event in events.keys()]
    assert values == targets

def test_get_extractor_by_name():
    tda = get_extractor('stFteXtrActOr')
    assert isinstance(tda, STFTExtractor)

def test_part_of_speech_extractor():
    stim = ComplexTextStim(join(TEXT_DIR, 'complex_stim_with_header.txt'))
    tl = stim.extract([PartOfSpeechExtractor()])
    df = tl.to_df()
    assert df.iloc[1, 3] == 'NN'
    assert df.shape == (4, 4)

def test_saliency_extractor():
        pytest.importorskip('cv2')
        image_dir = join(_get_test_data_path(), 'image')
        stim = ImageStim(join(image_dir, 'apple.jpg'))
        tl = stim.extract([SaliencyExtractor()])
        ms = tl.data['SaliencyExtractor'].data['max_saliency']
        assert np.isclose(ms,0.99669953)
        sf = tl.data['SaliencyExtractor'].data['frac_high_saliency']
        assert np.isclose(sf,0.27461971)

def test_tesseract_extractor():
    pytest.importorskip('pytesseract')
    image_dir = join(_get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'button.jpg'))
    ext = TesseractExtractor()
    output = ext.apply(stim).data['text']
    assert output == 'Exit'

@pytest.mark.skipif("'CLARIFAI_APP_ID' not in os.environ")
def test_clarifaiAPI_extractor():
    image_dir = join(_get_test_data_path(), 'image')
    stim = ImageStim(join(image_dir, 'apple.jpg'))
    ext = ClarifaiAPIExtractor()
    output = ext.apply(stim).data['tags']
    # Check success of request
    assert output['status_code'] == 'OK'
    # Check success of each image tagged
    for result in output['results']:
        assert result['status_code'] == 'OK'
        assert result['result']['tag']['classes']

@pytest.mark.skipif("'INDICO_APP_KEY' not in os.environ")
def test_indicoAPI_extractor():
    srtfile = join(_get_test_data_path(), 'text', 'wonderful.srt')
    srt_stim = ComplexTextStim(srtfile)
    ext = IndicoAPIExtractor(api_key=os.environ['INDICO_APP_KEY'],model = 'emotion')
    output = ext.apply(srt_stim)
    outdfKeys = set(output.to_df()['name'])
    outdfKeysCheck = set([
        'emotion_anger',
        'emotion_fear',
        'emotion_joy',
        'emotion_sadness',
        'emotion_surprise'])
    assert outdfKeys == outdfKeysCheck
