''' Converter classes that operate on VideoStim inputs. '''

from pliers.stimuli.video import VideoStim
from pliers.stimuli.audio import AudioStim
from .base import Converter


class VideoToAudioConverter(Converter):

    ''' Convert a VideoStim to an AudioStim by extracting the audio track
    using moviepy. '''
    _input_type = VideoStim
    _output_type = AudioStim
    VERSION = '1.0'

    def _convert(self, video):
        fps = AudioStim.get_sampling_rate(video.filename)
        return AudioStim(sampling_rate=fps,
                         clip=video.clip.audio)
