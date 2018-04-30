''' Filters that operate on TextStim inputs. '''

from .base import Filter, TemporalTrimmingFilter
from pliers.stimuli import (AudioStim,
                            TextStim,
                            ComplexTextStim,
                            TranscribedAudioCompoundStim)
from pliers.utils import attempt_to_import, verify_dependencies

import os
import tempfile

aeneas = attempt_to_import('aeneas')


class AudioFilter(Filter):

    ''' Base class for all audio filters. '''

    _input_type = AudioStim


class AudioTrimmingFilter(TemporalTrimmingFilter, AudioFilter):
    pass


class TranscribedAudioFilter(Filter):

    ''' Base class for all transcribed audio filters. '''

    _input_type = TranscribedAudioCompoundStim


class AeneasForcedAlignmentFilter(TranscribedAudioFilter):

    ''' Performs alignment of spoken text fragments within an audio file.
    Requires eSpeak to work, in addition to all Python dependencies. '''

    def _parse_sync_map(self, sync_map):
        verify_dependencies(['aeneas'])
        from aeneas.syncmap.fragment import SyncMapFragment

        texts = []
        for fragment in sync_map.fragments:
            if fragment.fragment_type == SyncMapFragment.REGULAR:
                offset = float(fragment.identifier)
                texts.append(TextStim(onset=float(fragment.begin) + offset,
                                      duration=float(fragment.length),
                                      text=fragment.text))
        return ComplexTextStim(elements=texts)

    def _filter(self, stim):
        verify_dependencies(['aeneas'])
        from aeneas.task import Task
        from aeneas.executetask import ExecuteTask

        audio_stim = stim.get_stim(AudioStim)
        text_path = tempfile.mktemp()
        with open(text_path, 'w') as f:
            cts = stim.get_stim(ComplexTextStim)
            offset = cts.onset if cts.onset else 0.0
            for txt in cts:
                f.write('%f|%s\n' % (offset, txt.text))

        with audio_stim.get_filename() as audio_path:
            config_string = 'task_language=eng|is_text_type=parsed'
            task = Task(config_string=config_string)
            task.audio_file_path_absolute = audio_path
            task.text_file_path_absolute = text_path

            ExecuteTask(task).execute()
            new_transcript = self._parse_sync_map(task.sync_map)

        os.remove(text_path)
        return TranscribedAudioCompoundStim(audio_stim, new_transcript)
