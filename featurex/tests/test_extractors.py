from unittest import TestCase
from os.path import join
from .utils import _get_test_data_path
from featurex.extractors.text import DictionaryExtractor, PartOfSpeechExtractor
from featurex.extractors.audio import STFTExtractor
from featurex.stimuli.text import ComplexTextStim
from featurex.stimuli.audio import AudioStim
from featurex.export import TimelineExporter
from featurex.extractors import get_extractor
from featurex.support.download import download_nltk_data

TEXT_DIR = join(_get_test_data_path(), 'text')


class TestExtractors(TestCase):

    def test_text_extractor(self):
        download_nltk_data()
        stim = ComplexTextStim(join(TEXT_DIR, 'sample_text.txt'),
                               columns='to', default_duration=1)
        td = DictionaryExtractor(join(TEXT_DIR, 'test_lexical_dictionary.txt'),
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

    def test_part_of_speech_extractor(self):
        stim = ComplexTextStim(join(TEXT_DIR, 'complex_stim_with_header.txt'))
        tl = stim.extract([PartOfSpeechExtractor()]).to_df()
        self.assertEquals(tl.iloc[1, 3], 'NN')
        self.assertEquals(tl.shape, (4, 4))
