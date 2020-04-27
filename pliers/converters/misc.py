"""Miscellaneous conversion classes."""

from pliers.extractors import ExtractorResult
from pliers.stimuli import SeriesStim
from .base import Converter


class ExtractorResultToSeriesConverter(Converter):
    """Converts an ExtractorResult instance to a list of SeriesStims."""

    _input_type = ExtractorResult
    _output_type = SeriesStim

    def _convert(self, result):
        df = result.to_df(timing=False, metadata=False, object_id=False)
        n_rows = df.shape[0]
        stims = []
        for i in n_rows:
            data = df.iloc[i, :]
            onset = result.onset[i]
            duration = result.duration[i]
            order = result.order[i]
            st = SeriesStim(data, onset=onset, duration=duration, order=order)
            stims.append(st)
        return stims
