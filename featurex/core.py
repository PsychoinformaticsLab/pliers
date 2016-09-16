from collections import OrderedDict, defaultdict
import pandas as pd
import numpy as np
from copy import deepcopy
from abc import ABCMeta, abstractmethod, abstractproperty


class Value(object):
    ''' The smallest unit of feature annotation. Binds a Stim and Extractor to
    one or more extracted values.
    Args:
        stim (Stim): The Stimulus associated with the extracted value.
        extractor (Extractor): The Extractor that produced the value.
        data (dict): The value(s) to register, where the dict keys are the
            feature names and the values are the values associated with those
            features.
        description (str): Optional description of the Value.
    Notes:
        Value instances are atemporal; they have no onset or duration
        information. To track temporal context, they must be attached to
        Events.
    '''
    def __init__(self, stim, extractor, data, description=None):

        self.stim = stim
        self.extractor = extractor
        self.data = data
        self.description = description


class Event(object):
    ''' A container for one or more Values; typically associated with a
    particular onset and duration.
    Args:
        onset (float): Onset of the Event relative to start of the associated
            stimulus.
        values (list): A list of Value instances associated with the Event.
        duration (float): The Duration of the Event.
    '''
    def __init__(self, onset=None, values=None, duration=None):

        self.onset = onset
        if values is None:
            values = []
        self.values = values
        self.duration = duration

    def add_value(self, value):
        ''' Add a new Value to the Event. '''
        self.values.append(value)


class Timeline(object):
    '''
    Args:
        events (list): A list of one or more Event instances to register on the
            timeline.
        period (float): The duration of each sampling frame within the
            timeline. If none is provided, all passed Events must have an onset
            property set, and the onsets will be used to register the Events
            at the appropriate point in the timeline. If a period value is
            passed, any events without an onset will be registered to the
            timeline based on their index (see Values).
    Notes:
        For any Event without an existing onset value, the onset will be
        set to equal (period * i), where i is the index of the current event
        within the passed list. This means that if a Timeline is initialized
        with Events that lack onsets, the list of Events should comprehensively
        cover the timeline, leaving no gaps (i.e., empty Events should be
        passed even when there are no relevant values to include).
    '''
    def __init__(self, events=None, period=None):

        self.events = OrderedDict()

        if events is not None:
            for i, ev in enumerate(events):
                onset = ev.onset
                if onset is None:
                    if period is None:
                        raise ValueError("onset attribute missing in at least "
                                         "one event, and no period argument was "
                                         "provided. If events do not all have "
                                         "valid onsets, you must explicitly"
                                         " specify the periodicity to assume.")
                    onset = period * i
                self.add_event(ev, onset, False)

        self._sort_events()
        self.period = period

    def _sort_events(self):
        ''' Sort the events OrderedDict by ascending onset. '''
        self.events = OrderedDict(sorted(self.events.items(),
                                         key=lambda t: t[0]))

    def add_event(self, event, onset=None, sort=True, merge=False):
        '''
        Register a new Event with the current timeline.
        Args:
            event (Event): The Event to add to the Timeline
            onset (float): The onset/onset at which to insert the Event.
                If None, the event must have a valid onset property.
            sort (bool): Whether or not to resort events by onset after adding.
            merge (bool): If True, all values in the passed event will be merged
                with any values in the existing event (with duplicates ignored).
                If False (default), new events will overwrite existing ones.
        '''
        if onset is None:
            onset = event.onset

        if onset is None:
            raise ValueError("If no onset is specified in add() call, "
                             "the passed Event must have a valid onset "
                             "attribute.")

        if onset in self.events and merge:
            event.values = list(set(event.values + self.events[onset].values))

        self.events[onset] = event

        if sort:
            self._sort_events()

    def merge(self, timeline):
        ''' Merge passed Timeline into current instance.
        Args:
            timeline (Timeline): the Timeline to merge with the current one.
        '''
        for event in timeline.events.keys():
            self.add_event(timeline.events[event], onset=event, merge=True)
            # TODO: handle potential period mismatches

    def to_df(self, format='long', extractor=False):
        ''' Return the Timeline as a pandas DataFrame.
        Args:
            format (str): Either 'long' (default) or 'wide'. In 'long' format,
            each row is a single sample for a single feature. In 'wide' format,
            samples are in rows and features are in columns.
            extractor (bool): If True, includes the name of the Extractor in
                the output (a separate column in the case of long format, and
                prepended to the column name in the case of wide).
        '''
        # local import to prevent circularity
        from .export import TimelineExporter
        return TimelineExporter.timeline_to_df(self, format, extractor)

    def dummy_code(self, string_only=True):
        ''' Returns a copy of the Timeline where all string variables (or all
        variables if string_only is False) are replaced with dummy-coded binary
        variables.

        Args:
            string_only (bool): If True (default), only string values are
                dummy-coded. If False, all values are replaced.

        Returns: A Timeline.

        Notes: Dummy variables are replaced by concatenating the original
            variable name with the original value. For example, if a variable
            is named 'A' and has values 'apple' and 'orange', the new variables
            will be named 'A_apple', 'A_orange', etc.
        '''
        # First pass: make dummies for all unique values of all variables
        dummies = defaultdict(lambda: defaultdict(dict))
        values = defaultdict(set)
        for onset, event in self.events.items():
            for value in event.values:
                for var, val in value.data.items():
                    dummies[var]['%s_%s' % (var, val)] = 0
                    values[var].add(val)
        dtypes = { k: pd.Series(list(v)).dtype for k, v in values.items() }

        # Second pass: replace string values with dummies
        result = deepcopy(self)
        for onset, event in result.events.items():
            for value in event.values:
                new_data = {}
                for var, val in value.data.items():
                    if dtypes[var] == np.object or not string_only:
                        _values = dummies[var].copy()
                        _values['%s_%s' % (var, val)] = 1
                        new_data.update(_values)
                    else:
                        new_data[var] = val
                value.data = new_data
        return result


class Transformer(object):

    __metaclass__ = ABCMeta

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self.name = name

    @abstractmethod
    def transform(self):
        pass

    @abstractproperty
    def target(self):
        pass

    @abstractproperty
    def __version__(self):
        pass
