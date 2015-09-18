from unittest import TestCase
from os.path import join
from .utils import _get_test_data_path
from featurex.extractors.text import TextDictionaryExtractor
from featurex.extractors.audio import STFTExtractor
from featurex.stimuli.text import ComplexTextStim
from featurex.stimuli.audio import AudioStim
from featurex.io import TimelineExporter
from featurex.extractors import get_extractor


class TestExtractors(TestCase):

    def test_text_extractor(self):
        text_dir = join(_get_test_data_path(), 'text')
        stim = ComplexTextStim(join(text_dir, 'sample_text.txt'),
                               columns='to', default_duration=1)
        td = TextDictionaryExtractor(join(text_dir,
                                          'test_lexical_dictionary.txt'),
                                     variables=['length', 'frequency'])
        self.assertEquals(td.data.shape, (7, 2))
        timeline = stim.extract([td])
        df = TimelineExporter.timeline_to_df(timeline)
        self.assertEquals(df.shape, (12, 4))
        self.assertEquals(df.iloc[9, 3], 10.6)

    def test_stft_extractor(self):
        audio_dir = join(_get_test_data_path(), 'audio')
        stim = AudioStim(join(audio_dir, 'barber.wav'))
        ext = STFTExtractor(frame_size=1., spectrogram=False,
                            bins=[(100, 300), (300, 3000), (3000, 20000)])
        timeline = stim.extract([ext])
        df = timeline.to_df('long')
        self.assertEquals(df.shape, (1671, 4))

    def test_get_extractor_by_name(self):
        tda = get_extractor('stFteXtrActOr')
        self.assertTrue(isinstance(tda, STFTExtractor))
