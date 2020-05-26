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
        if 'object_id' in df.columns:
            df = df.drop(['object_id'], axis=1)
        stims = []
        for i, data in df.iterrows():
            onset = result.onset[i] if result.onset is not None else None
            dur = result.duration[i] if result.duration is not None else None
            order = result.order[i] if result.order is not None else i
            st = SeriesStim(data, onset=onset, duration=dur, order=order)
            stims.append(st)
        return stims
