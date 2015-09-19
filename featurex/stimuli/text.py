from featurex.stimuli import Stim
from featurex.core import Timeline, Event
import pandas as pd


class TextStim(Stim):

    ''' Any text stimulus. '''

    def __init__(self, text):
        self.text = text


class DynamicTextStim(TextStim):

    ''' A text stimulus with timing/onset information. '''

    def __init__(self, text, order, onset=None, duration=None):
        self.order = order
        self.onset = onset
        self.duration = duration
        super(DynamicTextStim, self).__init__(text)


class ComplexTextStim(object):

    ''' A collection of text stims (e.g., a story), typically ordered and with
    onsets and/or durations associated with each element.
    Args:
        filename (str): The filename to read from. Must be tab-delimited text.
            Files must always contain a column containing the text of each
            stimulus in the collection. Optionally, additional columns can be
            included that contain duration and onset information. If a header
            row is present in the file, valid columns must be labeled as
            'text', 'onset', and 'duration' where available (though only text
            is mandatory). If no header is present in the file, the columns
            argument will be used to infer the indices of the key columns.
        columns (str): Optional specification of column order. An abbreviated
            string denoting the column position of text, onset, and duration
            in the file. Use t for text, o for onset, d for duration. For
            example, passing 'ot' indicates that the first column contains
            the onsets and the second contains the text. Passing 'tod'
            indicates that the first three columns contain text, onset, and
            duration information, respectively. Note that if the input file
            contains a header row, the columns argument will be ignored.
        default_duration (float): the duration to assign to any text elements
            in the collection that do not have an explicit value provided
            in the input file.
    '''

    def __init__(self, filename, columns='tod', default_duration=None):

        self.elements = []
        tod_names = {'t': 'text', 'o': 'onset', 'd': 'duration'}

        first_row = open(filename).readline().strip().split('\t')
        if len(set(first_row) & set(tod_names.values())):
            col_names = None
        else:
            col_names = [tod_names[x] for x in columns]

        data = pd.read_csv(filename, sep='\t', names=col_names)

        for i, r in data.iterrows():
            if 'onset' not in r:
                elem = TextStim(r['text'])
            else:
                duration = r.get('duration', None)
                if duration is None:
                    duration = default_duration
                elem = DynamicTextStim(r['text'], i, r['onset'], duration)
            self.elements.append(elem)

    def __iter__(self):
        """ Iterate text elements. """
        for elem in self.elements:
            yield elem

    def extract(self, extractors, merge_events=True):
        timeline = Timeline()
        for ext in extractors:
            if ext.target.__name__ == self.__class__.__name__:
                events = ext.apply(self)
                for ev in events:
                    timeline.add_event(ev, merge=merge_events)
            else:
                for elem in self.elements:
                    event = Event(onset=elem.onset)
                    event.add_value(ext.apply(elem))
                    timeline.add_event(event, merge=merge_events)
        return timeline

    @classmethod
    def from_text(cls, text, unit='word'):
        """ Initialize from a single string, by automatically segmenting into
        individual strings.
        """
        pass
