from pliers.converters import VideoToAudioConverter
from pliers.filters import TemporalTrimmingFilter, PunctuationRemovalFilter, LowerCasingFilter, TokenizingFilter
from pliers.filters.text import TextFilter
from pliers.filters.audio import AeneasForcedAlignmentFilter
from pliers.stimuli import AudioStim, TextStim, ComplexTextStim, TranscribedAudioCompoundStim, VideoStim
from pliers.graph import Graph
import re

class SpeakerFilter(TextFilter):

    def _filter(self, stim):
        filtered = stim.text
        matches = re.findall('\(\s*[A-Z]{2,}\s*\)', filtered)
        matches.extend(re.findall('[A-Z]{2,}:', filtered))
        for m in matches:
            filtered = filtered.replace(m, '')
        return TextStim(text=filtered)


def align_transcript(audio_file, transcript_file)
    STIM_DIR = '/Users/quinnmac/Documents/NeuralComputation/project/stims/'

    # Audio preprocessing
    vid = VideoStim(STIM_DIR + 'Merlin.mp4')
    aud = VideoToAudioConverter().transform(vid)
    aud = TemporalTrimmingFilter(start=40.5).transform(aud)
    print('Done with audio preprocessing, duration: ' + str(aud.duration))

    # Transcript preprocessing
    txt = ComplexTextStim(STIM_DIR + 'transcription/subtitles.srt')
    txt_preproc = Graph()
    txt_preproc.add_nodes([SpeakerFilter(), LowerCasingFilter(), PunctuationRemovalFilter()],  mode='vertical')
    txt = ComplexTextStim(elements=txt_preproc.transform(txt, merge=False))
    print('Done with text preprocessing')

    # Perform alignment
    aligner = AeneasForcedAlignmentFilter()
    if word_level:
        final_transcript = []
        for fragment in txt:
            if fragment.text == '':
                continue
            end = fragment.onset + fragment.duration
            if end > aud.duration:
                break
            crop_filter = TemporalTrimmingFilter(start=fragment.onset, end=end)
            audio_segment = crop_filter.transform(aud)
            fragment.onset = 0.0
            words = ComplexTextStim(onset=crop_filter.start, elements=TokenizingFilter().transform(fragment))
            stim = TranscribedAudioCompoundStim(audio_segment, words)
            new_transcribed = aligner.transform(stim)
            final_transcript.extend(new_transcribed.get_stim(ComplexTextStim).elements)
        result = ComplexTextStim(elements=final_transcript)
    else:
        stim = TranscribedAudioCompoundStim(aud, txt)
        result = aligner.transform(stim).get_stim(ComplexTextStim)

    result.save(STIM_DIR + 'transcription/aeneas_exact_transcript.txt')
    result.to_srt(STIM_DIR + 'transcription/aeneas_exact_transcript.srt')
    print('Done with forced alignment')

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python aeneas_align.py <audio_file> <srt_file> <word_leve>')
