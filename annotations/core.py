from collections import OrderedDict


class Note(object):
    ''' The smallest unit of annotation. Binds a Stim and Annotator to one or
    more extracted values.
    Args:
        stim (Stim): The Stimulus associated with the annotated value.
        annotator (Annotator): The Annotator that produced the value.
        data (dict): The value(s) to register, where the dict keys are the
            feature names and the values are the values associated with those
            features.
        description (str): Optional description of the Note.
    Notes:
        Note instances are atemporal; they have no onset or duration
        information. To track temporal context, they must be attached to
        Events.
    '''
    def __init__(self, stim, annotator, data, description=None):

        self.stim = stim
        self.annotator = annotator
        self.data = data
        self.description = description


class Event(object):
    ''' A container for one or more Notes; typically associated with a
    particular onset and duration.
    Args:
        onset (float): Onset of the Event relative to start of the associated
            stimulus.
        notes (list): A list of Note instances associated with the Event.
        duration (float): The Duration of the Event.
    '''
    def __init__(self, onset=None, notes=None, duration=None):

        self.onset = onset
        if notes is None:
            notes = []
        self.notes = notes
        self.duration = duration

    def add_note(self, note):
        ''' Add a new Note to the Event. '''
        self.notes.append(note)


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
            timeline based on their index (see Notes).
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

    def merge(self, timeline):
        ''' Merge passed Timeline into current instance.
        Args:
            timeline (Timeline): the Timeline to merge with the current one.
        '''
        for event in timeline.events:
            self.add_event(event, merge=True)
            # TODO: handle potential period mismatches

    def to_df(self, format='long'):
        ''' Return the Timeline as a pandas DataFrame.
        Args:
            format (str): Either 'long' (default) or 'wide'. In 'long' format,
            each row is a single sample for a single feature. In 'wide' format,
            samples are in rows and features are in columns.
        '''
        # local import to prevent circularity
        from .io import TimelineExporter
        return TimelineExporter.timeline_to_df(self, format)
