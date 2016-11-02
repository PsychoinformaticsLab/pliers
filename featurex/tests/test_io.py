from .utils import get_test_data_path
from featurex.stimuli import load_stims
from featurex.stimuli.audio import AudioStim
from featurex.extractors.audio import STFTExtractor
from featurex.export import convert_to_long_format
from os.path import join
from six import string_types


def test_magic_loader():
    text_file = join(get_test_data_path(), 'text', 'sample_text.txt')
    audio_file = join(get_test_data_path(), 'audio', 'barber.wav')
    video_file = join(get_test_data_path(), 'video', 'small.mp4')
    stim_files = [text_file, audio_file, video_file]
    stims = load_stims(stim_files)
    assert len(stims) == 3
    assert round(stims[1].duration) == 57
    assert isinstance(stims[0].text, string_types)
    assert stims[2].width == 560


def test_convert_to_long():
    audio_dir = join(get_test_data_path(), 'audio')
    stim = AudioStim(join(audio_dir, 'barber.wav'))
    ext = STFTExtractor(frame_size=1., spectrogram=False,
                        bins=[(100, 300), (300, 3000), (3000, 20000)])
    # This probably doesn't work if doing stim.extract() due to multi-level
    timeline = ext.extract(stim).to_df(stim_name=True)
    assert '100_300' in timeline.columns
    long_timeline = convert_to_long_format(timeline)
    assert long_timeline.shape == (timeline.shape[0] * 3, 5)
    assert 'feature' in long_timeline.columns
    assert 'value' in long_timeline.columns
    assert '100_300' not in long_timeline.columns
