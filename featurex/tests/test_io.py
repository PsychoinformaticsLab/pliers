from .utils import _get_test_data_path
from featurex.stimuli import load_stims
from featurex.stimuli.audio import AudioStim
from featurex.extractors.audio import STFTExtractor
from featurex import Value, Event, Timeline
from os.path import join
from six import string_types


def test_magic_loader():
    text_file = join(_get_test_data_path(), 'text', 'sample_text.txt')
    audio_file = join(_get_test_data_path(), 'audio', 'barber.wav')
    video_file = join(_get_test_data_path(), 'video', 'small.mp4')
    stim_files = [text_file, audio_file, video_file]
    stims = load_stims(stim_files)
    assert len(stims) == 3
    assert round(stims[1].duration) == 57
    assert isinstance(stims[0].text, string_types)
    assert stims[2].width == 560

def test_timeline_export():
    audio_dir = join(_get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'barber.wav'))
    ext = STFTExtractor(frame_size=1., spectrogram=False,
                        bins=[(100, 300), (300, 3000), (3000, 20000)])
    timeline = stim.extract([ext])
    df = timeline.to_df(format='wide', extractor=True)
    assert len(df.columns.levels) == 3
    df = timeline.to_df(format='wide', extractor=False)
    assert len(df.columns.levels) == 2