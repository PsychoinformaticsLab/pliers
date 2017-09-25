''' Converter classes that operate on VideoStim inputs. '''

from pliers.stimuli.video import VideoStim
from pliers.stimuli.audio import AudioStim
from .base import Converter


class VideoToAudioConverter(Converter):

    ''' Convert a VideoStim to an AudioStim by extracting the audio track
    using moviepy. '''
    _input_type = VideoStim
    _output_type = AudioStim

    def _convert(self, video):
        return AudioStim(clip=video.clip.audio, onset=video.onset)
