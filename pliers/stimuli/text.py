''' Classes that represent text or sequences of text. '''

import re
import pandas as pd
from six import string_types
from six.moves.urllib.request import urlopen
from pliers.support.decorators import requires_nltk_corpus
from .base import Stim, CollectionStimMixin


class TextStim(Stim):

    ''' Any simple text stimulus--most commonly a single word.
    Args:
        filename (str): Path to input file, if one exists.
        text (str): Text value to store. If none is provided, value is read
            from filename.
        onset (float): Optional onset of the text presentation (in secs) with
            respect to some more general context or timeline the user wishes
            to keep track of.
        duration (float): Optional duration of the TextStim, in seconds.
    '''

    def __init__(self, filename=None, text=None, onset=None, duration=None, url=None):
        if filename is not None and text is None:
            text = open(filename).read()
        if url is not None:
            text = urlopen(url).read()
        self.text = text
        name = 'text[%s]' % text[:40]  # Truncate at 40 chars
        super(TextStim, self).__init__(filename, onset, duration, name)


class ComplexTextStim(Stim, CollectionStimMixin):

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
        onset (float): Optional onset of the ComplexTextStim relative to some
            more general context.
        duration (float): Optional duration of the ComplexTextStim withing some
            more general context.
        columns (str): Optional specification of column order. An abbreviated
            string denoting the column position of text, onset, and duration
            in the file. Use t for text, o for onset, d for duration. For
            example, passing 'ot' indicates that the first column contains
            the onsets and the second contains the text. E.g., passing 'tod'
            indicates that the first three columns contain text, onset, and
            duration information, respectively. Note that if the input file
            contains a header row, the columns argument will be ignored.
        default_duration (float): the duration to assign to any text elements
            in the collection that do not have an explicit value provided
            in the input file.
        elements (list): An optional list of TextStims that comprise the
            ComplexTextStim. If both the filename and elements arguments are
            passed, the TextStims in elements will be appended to the ones
            extracted from the file.
        text (str): Optional multi-token string to convert to a ComplexTextStim.
        unit (str): The unit of segmentation. Either 'word' or 'sentence'.
            Ignored if text is None.
        tokenizer: Optional tokenizer to use if initializing from text. If
            passed, will override the default nltk tokenizers. If a string is
            passed, it is interpreted as a capturing regex and passed to
            re.findall(). Otherwise, must be an object that implements a
            tokenize() method and returns a list of tokens. Ignored if text is
            None.
        language (str): The language to use; passed to nltk. Only used if
            tokenizer is None. Defaults to English. Ignored if text is None.
    '''

    def __init__(self, filename=None, onset=None, duration=None, columns=None,
                 default_duration=None, elements=None, text=None, unit='word',
                 tokenizer=None, language='english'):

        if filename is None and elements is None and text is None:
            raise ValueError("At least one of the 'filename', 'elements', or "
                             "text arguments must be specified.")

        self._elements = []

        if filename is not None:
            if filename.endswith("srt"):
                self._from_srt(filename)
            else:
                self._from_file(filename, columns, default_duration)

        if elements is not None:
            self._elements.extend(elements)

        if text is not None:
            self._from_text(text, unit, tokenizer, language)

        super(ComplexTextStim, self).__init__(filename, onset, duration)

    @property
    def elements(self):
        return self._elements

    def _from_file(self, filename, columns, default_duration):
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
                elem = TextStim(filename, r['text'], r['onset'], duration)
            self._elements.append(elem)

    def _from_srt(self, filename):
        import pysrt

        data = pysrt.open(filename)
        list_ = [[] for _ in data]
        for i, row in enumerate(data):
            start = tuple(row.start)
            start_time = self._to_sec(start)

            end_ = tuple(row.end)
            duration = self._to_sec(end_) - start_time

            line = re.sub('\s+', ' ', row.text)
            list_[i] = [line, start_time, duration]

        # Convert to pandas DataFrame
        df = pd.DataFrame(columns=["text", "onset", "duration"], data=list_)

        for i, r in df.iterrows():
            elem = TextStim(filename, r['text'], r['onset'], r["duration"])
            self._elements.append(elem)

    def __iter__(self):
        """ Iterate text elements. """
        for elem in self._elements:
            yield elem

    def _to_sec(self, tup):
        hours, mins, secs, msecs = tup
        total_msecs = (hours * 60 * 60 * 1000) + (mins * 60 * 1000) + \
            (secs * 1000) + msecs
        total_secs = total_msecs / 1000.
        return total_secs

    def _from_text(self, text, unit, tokenizer, language):

        if tokenizer is not None:
            if isinstance(tokenizer, string_types):
                tokens = re.findall(tokenizer, text)
            else:
                tokens = tokenizer.tokenize(text)
        else:
            import nltk

            @requires_nltk_corpus
            def tokenize_text(text):
                if unit == 'word':
                    return nltk.word_tokenize(text, language)
                elif unit.startswith('sent'):
                    return nltk.sent_tokenize(text, language)
                else:
                    raise ValueError(
                        "unit must be either 'word' or 'sentence'")

            tokens = tokenize_text(text)

        # TODO: track order as a separate attribute from duration, because we
        # can't treat serial position as if it were time in seconds.
        for i, t in enumerate(tokens):
            self._elements.append(TextStim(text=t, onset=None, duration=None))
