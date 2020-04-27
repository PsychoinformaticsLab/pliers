"""Miscellaneous conversion classes."""

from pliers.extractors import ExtractorResult
from pliers.stimuli import DataFrameStim
from .base import Converter


class ExtractorResultToDFConverter(Converter):

    _input_type = ExtractorResult
    _output_type = DataFrameStim

    def _convert(self, result):
        pass
