from pliers.filters import (TemporalTrimmingFilter,
                            PunctuationRemovalFilter,
                            LowerCasingFilter,
                            TokenizingFilter,
                            AeneasForcedAlignmentFilter)
from pliers.filters.text import TextFilter
from pliers.stimuli import (AudioStim,
                            TextStim,
                            ComplexTextStim,
                            TranscribedAudioCompoundStim)
from pliers.graph import Graph
import re
import sys


class SpeakerFilter(TextFilter):

    def _filter(self, stim):
        filtered = stim.text
        matches = re.findall('\(\s*[A-Z]{2,}\s*\)', filtered)
        matches.extend(re.findall('[A-Z]{2,}:', filtered))
        for m in matches:
            filtered = filtered.replace(m, '')
        return TextStim(text=filtered)


def preprocess_text(transcript_file):
    # Transcript preprocessing
    txt = ComplexTextStim(transcript_file)
    txt_preproc = Graph()
    txt_preproc.add_nodes([SpeakerFilter(),
                           LowerCasingFilter(),
                           PunctuationRemovalFilter()],
                          mode='vertical')
    txt = ComplexTextStim(elements=txt_preproc.transform(txt, merge=False))
    return txt

def align_transcript(audio_file, transcript_file, output_file, level='word',
                     preprocess=True):
    # Load stimuli
    aud = AudioStim(audio_file)
    if preprocess:
        txt = preprocess_text(transcript_file)
    else:
        txt = ComplexTextStim(transcript_file)

    # Perform alignment
    aligner = AeneasForcedAlignmentFilter()
    if level == 'word':
        # Segment the audio file into phrase chunks, align words within each chunk
        final = []
        for fragment in txt:
            if fragment.text == '':
                continue
            end = fragment.onset + fragment.duration
            if end > aud.duration:
                break
            crop_filter = TemporalTrimmingFilter(start=fragment.onset,
                                                 end=end)
            audio_segment = crop_filter.transform(aud)
            fragment.onset = 0.0
            words = ComplexTextStim(onset=crop_filter.start,
                                    elements=TokenizingFilter().transform(fragment))
            stim = TranscribedAudioCompoundStim(audio_segment, words)
            new_transcribed = aligner.transform(stim)
            final.extend(new_transcribed.get_stim(ComplexTextStim).elements)
        result = ComplexTextStim(elements=final)
    else:
        stim = TranscribedAudioCompoundStim(aud, txt)
        result = aligner.transform(stim).get_stim(ComplexTextStim)

    result.save(output_file)

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage: python aeneas_align.py <audio_file> <srt_file> <output_srt_path> <level>')

    align_transcript(*sys.argv[1:])
    print('Done with forced alignment')
