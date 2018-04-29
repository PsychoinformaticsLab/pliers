''' Filters that operate on TextStim inputs. '''

from .base import Filter, TemporalTrimmingFilter
from pliers.stimuli import AudioStim, TextStim, ComplexTextStim, TranscribedAudioCompoundStim
import aeneas
from aeneas.executetask import ExecuteTask
from aeneas.task import Task
from aeneas.runtimeconfiguration import RuntimeConfiguration
import os
import tempfile


class AudioFilter(Filter):

    ''' Base class for all audio filters. '''

    _input_type = AudioStim


class AudioTrimmingFilter(TemporalTrimmingFilter, AudioFilter):
    pass


class TranscribedAudioFilter(Filter):

    _input_type = (AudioStim, ComplexTextStim)


class AeneasForcedAlignmentFilter(TranscribedAudioFilter):

    def _parse_sync_map(self, sync_map):
        texts = []
        for fragment in sync_map.fragments:
            if fragment.fragment_type == 0:#aeneas.syncmap.fragment.SyncMapFragment.REGULAR:
                offset = float(fragment.identifier)
                texts.append(TextStim(onset=float(fragment.begin) + offset,
                                      duration=float(fragment.length),
                                      text=fragment.text))
        return ComplexTextStim(elements=texts)

    def _filter(self, stim):
        audio_stim = stim.get_stim(AudioStim)
        text_path = tempfile.mktemp()
        with open(text_path, 'w') as f:
            cts = stim.get_stim(ComplexTextStim)
            offset = cts.onset if cts.onset else 0.0
            for txt in cts:
                f.write('%f|%s\n' % (offset, txt.text))

        with audio_stim.get_filename() as audio_path:
            rconf = {RuntimeConfiguration.MFCC_MASK_NONSPEECH: True}
            config_string = 'task_language=eng|is_text_type=parsed'
            task = Task(config_string=config_string, rconf=rconf)
            task.audio_file_path_absolute = audio_path
            task.text_file_path_absolute = text_path

            try:
                ExecuteTask(task).execute()
            except aeneas.executetask.ExecuteTaskExecutionError:
                print('error')

            new_transcript = self._parse_sync_map(task.sync_map)

        os.remove(text_path)
        return TranscribedAudioCompoundStim(audio_stim, new_transcript)
