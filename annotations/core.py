# import pandas as pd
from collections import OrderedDict


class Note(object):

    def __init__(self, stim, annotator, data, description=None):

        self.stim = stim
        self.annotator = annotator
        self.data = data
        self.description = description


class Event(object):

    def __init__(self, position=None, notes=None, duration=None):

        self.position = position
        if notes is None:
            notes = []
        self.notes = notes
        self.duration = duration

    def add_note(self, note):
        self.notes.append(note)


class Timeline(object):

    def __init__(self, events=None, period=None):

        self.events = OrderedDict()

        if events is not None:
            for i, ev in enumerate(events):
                position = ev.position
                if position is None:
                    if period is None:
                        raise ValueError("Position attribute missing in at least "
                                         "one event, and no period argument was "
                                         "provided. If events do not all have "
                                         "valid positions, you must explicitly"
                                         " specify the periodicity to assume.")
                    position = period * i
                self.add_event(ev, position, False)

        self._sort_events()
        self.period = period

    def _sort_events(self):
        ''' Sort the events OrderedDict by ascending position. '''
        self.events = OrderedDict(sorted(self.events.items(),
                                         key=lambda t: t[0]))

    def add_event(self, event, position=None, sort=True, merge=False):
        '''
        Args:
            event (Event): The Event to add to the Timeline
            position (float): The position/onset at which to insert the Event.
                If None, the event must have a valid position property.
            sort (bool): Wether or not to sort events by position after adding.
            merge (bool): If True, all notes in the passed event will be merged
                with any notes in the existing event (with duplicates ignored).
                If False (default), new events will overwrite existing ones.
        '''
        if position is None:
            position = event.position

        if position is None:
            raise ValueError("If no position is specified in add() call, "
                             "the passed Event must have a valid position "
                             "attribute.")

        if position in self.events and merge:
            event.notes = list(set(event.notes + self.events[position].notes))

        self.events[position] = event

        if sort:
            self._sort_events()
