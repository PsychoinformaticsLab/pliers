from os.path import join

import pytest

from pliers.tests.utils import get_test_data_path
from pliers.stimuli import load_stims

def test_magic_loader():
    text_file = join(get_test_data_path(), 'text', 'sample_text.txt')
    audio_file = join(get_test_data_path(), 'audio', 'barber.wav')
    video_file = join(get_test_data_path(), 'video', 'small.mp4')
    stim_files = [text_file, audio_file, video_file]
    stims = load_stims(stim_files)
    assert len(stims) == 3
    assert round(stims[1].duration) == 57
    assert isinstance(stims[0].text, str)
    assert stims[2].width == 560


def test_magic_loader2():
    text_file = join(get_test_data_path(), 'text', 'sample_text.txt')
    video_url = 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4'
    audio_url = 'https://www2.cs.uic.edu/~i101/SoundFiles/Fanfare60.wav'
    image_url = 'https://storage.googleapis.com/gtv-videos-bucket/sample/images/ElephantsDream.jpg'
    text_url = 'https://github.com/psychoinformaticslab/pliers/blob/master/README.rst'

    stims = load_stims([text_file, video_url, audio_url, image_url, text_url])
    assert len(stims) == 5
    assert stims[1].fps == 30.0
    assert stims[3].data.shape == (288, 360, 3)


def test_loader_nonexistent():
    text_file = 'this/doesnt/exist.txt'
    with pytest.raises(IOError):
        stims = load_stims(text_file)

    audio_file = 'no/audio/here.wav'
    with pytest.raises(IOError):
        stims = load_stims([text_file, audio_file])

    text_file = join(get_test_data_path(), 'text', 'sample_text.txt')
    stims = load_stims([text_file, audio_file], fail_silently=True)
    assert len(stims) == 1

    with pytest.raises(IOError):
        stims = load_stims(audio_file, fail_silently=True)
