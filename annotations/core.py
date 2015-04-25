# import pandas as pd
from collections import OrderedDict


class Note(object):

    def __init__(self, stim, annotator, data, description=None):

        self.stim = stim
        self.annotator = annotator
        self.data = data
        self.description = description


class Event(object):

    def __init__(self, onset=None, notes=None, duration=None):

        self.onset = onset
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
        Args:
            event (Event): The Event to add to the Timeline
            onset (float): The onset/onset at which to insert the Event.
                If None, the event must have a valid onset property.
            sort (bool): Wether or not to sort events by onset after adding.
            merge (bool): If True, all notes in the passed event will be merged
                with any notes in the existing event (with duplicates ignored).
                If False (default), new events will overwrite existing ones.
        '''
        if onset is None:
            onset = event.onset

        if onset is None:
            raise ValueError("If no onset is specified in add() call, "
                             "the passed Event must have a valid onset "
                             "attribute.")

        if onset in self.events and merge:
            event.notes = list(set(event.notes + self.events[onset].notes))

        self.events[onset] = event

        if sort:
            self._sort_events()
