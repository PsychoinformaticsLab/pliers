from unittest import TestCase
from os.path import join
import os
from .utils import _get_test_data_path
from featurex.extractors.text import (DictionaryExtractor,
                                      PartOfSpeechExtractor,
                                      PredefinedDictionaryExtractor)
from featurex.extractors.audio import STFTExtractor, MeanAmplitudeExtractor
from featurex.extractors.api import ClarifaiAPIExtractor
from featurex.stimuli.text import ComplexTextStim
from featurex.stimuli.video import ImageStim
from featurex.stimuli.audio import AudioStim, TranscribedAudioStim
from featurex.export import TimelineExporter
from featurex.extractors import get_extractor
from featurex.support.download import download_nltk_data
import numpy as np

TEXT_DIR = join(_get_test_data_path(), 'text')


class TestExtractors(TestCase):

    @classmethod
    def setUpClass(self):

        download_nltk_data()

    def test_check_target_type(self):
        audio_dir = join(_get_test_data_path(), 'audio')
        stim = AudioStim(join(audio_dir, 'barber.wav'))
        td = DictionaryExtractor(join(TEXT_DIR, 'test_lexical_dictionary.txt'),
                                 variables=['length', 'frequency'])
        with self.assertRaises(TypeError):
            stim.extract([td])

    def test_text_extractor(self):
        stim = ComplexTextStim(join(TEXT_DIR, 'sample_text.txt'),
                               columns='to', default_duration=1)
        td = DictionaryExtractor(join(TEXT_DIR, 'test_lexical_dictionary.txt'),
                                 variables=['length', 'frequency'])
        self.assertEquals(td.data.shape, (7, 2))
        timeline = stim.extract([td])
        df = timeline.to_df()
        self.assertTrue(np.isnan(df.iloc[0, 3]))
        self.assertEquals(df.shape, (12, 4))
        target = df.query('name=="frequency" & onset==5')['value'].values
        self.assertEquals(target, 10.6)

    def test_predefined_dictionary_extractor(self):
        text = """enormous chunks of ice that have been frozen for thousands of
                  years are breaking apart and melting away"""
        stim = ComplexTextStim.from_text(text)
        td = PredefinedDictionaryExtractor(['aoa/Freq_pm', 'affect/V.Mean.Sum'])
        timeline = stim.extract([td])
        df = TimelineExporter.timeline_to_df(timeline)
        self.assertEqual(df.shape, (36, 4))
        valid_rows = df.query('name == "affect_V.Mean.Sum"').dropna()
        self.assertEqual(len(valid_rows), 3)

    def test_stft_extractor(self):
        audio_dir = join(_get_test_data_path(), 'audio')
        stim = AudioStim(join(audio_dir, 'barber.wav'))
        ext = STFTExtractor(frame_size=1., spectrogram=False,
                            bins=[(100, 300), (300, 3000), (3000, 20000)])
        timeline = stim.extract([ext])
        df = timeline.to_df('long')
        self.assertEquals(df.shape, (1671, 4))
    
    def test_mean_amplitude_extractor(self):
        audio_dir = join(_get_test_data_path(), 'audio')
        text_dir = join(_get_test_data_path(), 'text')
        stim = TranscribedAudioStim(join(audio_dir, "barber_edited.wav"),
                                    join(text_dir, "wonderful_edited.srt"))
        ext = MeanAmplitudeExtractor()
        timeline = stim.extract([ext])
        targets = [100., 150.]
        events = timeline.events
        values = [events[event].values[0].data["mean_amplitude"] for event in events.keys()]
        self.assertEquals(values, targets)

    def test_get_extractor_by_name(self):
        tda = get_extractor('stFteXtrActOr')
        self.assertTrue(isinstance(tda, STFTExtractor))

    def test_part_of_speech_extractor(self):
        stim = ComplexTextStim(join(TEXT_DIR, 'complex_stim_with_header.txt'))
        tl = stim.extract([PartOfSpeechExtractor()])
        df = tl.to_df()
        self.assertEquals(df.iloc[1, 3], 'NN')
        self.assertEquals(df.shape, (4, 4))

    def test_api_extractor(self):
        image_dir = join(_get_test_data_path(), 'image')
        stim = ImageStim(join(image_dir, 'apple.jpg'))
        if 'CLARIFAI_APP_ID' in os.environ:
            ext = ClarifaiAPIExtractor()
            output = ext.apply(stim).data['tags']
            # Check success of request
            self.assertEquals(output['status_code'], 'OK')
            # Check success of each image tagged
            for result in output['results']:
                self.assertEquals(result['status_code'], 'OK')
                self.assertTrue(result['result']['tag']['classes'])
