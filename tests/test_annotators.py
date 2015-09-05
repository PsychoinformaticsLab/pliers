from unittest import TestCase
from os.path import join
from .utils import _get_test_data_path
from annotations.annotators.text import TextDictionaryAnnotator
from annotations.annotators.audio import STFTAnnotator
from annotations.stims import ComplexTextStim, AudioStim
from annotations.io import TimelineExporter


class TestAnnotations(TestCase):

    def test_text_annotator(self):
        text_dir = join(_get_test_data_path(), 'text')
        stim = ComplexTextStim(join(text_dir, 'sample_text.txt'),
                               columns='to', default_duration=1)
        td = TextDictionaryAnnotator(join(text_dir,
                                          'test_lexical_dictionary.txt'),
                                     variables=['length', 'frequency'])
        self.assertEquals(td.data.shape, (7, 2))
        timeline = stim.annotate([td])
        df = TimelineExporter.timeline_to_df(timeline)
        self.assertEquals(df.shape, (12, 4))
        self.assertEquals(df.iloc[9, 3], 10.6)

    def test_stft_annotator(self):
        audio_dir = join(_get_test_data_path(), 'audio')
        stim = AudioStim(join(audio_dir, 'barber.wav'))
        ann = STFTAnnotator(frame_size=1., spectrogram=False,
                            bins=[(100, 300), (300, 3000), (3000, 20000)])
        timeline = stim.annotate([ann])
        df = timeline.to_df('long')
        self.assertEquals(df.shape, (1671, 4))
