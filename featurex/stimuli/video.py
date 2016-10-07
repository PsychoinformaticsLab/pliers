from __future__ import division
from featurex.stimuli import DynamicStim
from featurex.stimuli.image import ImageStim
from featurex.core import Timeline, Event
from moviepy.video.io.VideoFileClip import VideoFileClip
import pandas as pd


class VideoFrameStim(ImageStim):

    ''' A single frame of video. '''

    def __init__(self, video, frame_num, duration=None, filename=None, data=None):
        super(VideoFrameStim, self).__init__(filename, data)
        self.video = video
        self.frame_num = frame_num
        spf = 1. / video.fps
        if duration is None:
            self.duration = spf
        else:
            self.duration = duration
        self.onset = frame_num * spf


class VideoStim(DynamicStim):

    ''' A video. '''

    def __init__(self, filename):

        self.clip = VideoFileClip(filename)
        self.fps = self.clip.fps
        self.width = self.clip.w
        self.height = self.clip.h

        self.frames = []
        self.frames = [f for f in self.clip.iter_frames()]
        self.n_frames = len(self.frames)

        super(VideoStim, self).__init__(filename)

    def _extract_duration(self):
        self.duration = self.n_frames * 1. / self.fps

    def __iter__(self):
        """ Frame iteration. """
        for i, f in enumerate(self.frames):
            yield VideoFrameStim(self, i, data=f)

    def extract(self, extractors, merge_events=True, **kwargs):
        period = 1. / self.fps
        timeline = Timeline(period=period)
        for ext in extractors:
            # For VideoExtractors, pass the entire stim
            if ext.target.__name__ == self.__class__.__name__:
                events = ext.transform(self, **kwargs)
                for ev in events:
                    timeline.add_event(ev, merge=merge_events)
            # Otherwise, for images, loop over frames
            else:
                c = 0
                for frame in self:
                    if frame.data is not None:
                        event = Event(onset=c * period)
                        event.add_value(ext.transform(frame))
                        timeline.add_event(event, merge=merge_events)
                        c += 1
        return timeline


class DerivedVideoStim(VideoStim):
    """
    VideoStim containing keyframes (for API calls). Each keyframe is associated
    with a duration reflecting the length of its "scene."
    """
    def __init__(self, filename, **kwargs):
        super(DerivedVideoStim, self).__init__(filename)
        self.history = pd.DataFrame(columns=["filter", "value", "n_frames"])
        self.tagged_frames = self.frames
        self.frame_index = range(len(self.frames))
        self._filter(**kwargs)
        
    def filter(self, **kwargs):
        self._filter(**kwargs)

    def _filter(self, every=None, hertz=None, keyframes=None):
        name = "None"
        thresh = 0
        new_idx = self.frame_index
        if every is not None:
            name = "every"
            thresh = every
            new_idx = range(self.n_frames)[::every]
        elif hertz is not None:
            name = "hertz"
            thresh = hertz
            interval = int(self.fps / hertz)
            new_idx = range(self.n_frames)[::interval]
        elif keyframes is not None:
            import cv2
            name = "keyframes"
            thresh = keyframes
            diffs = []
            for i, img in enumerate(self.frames):
                if i == 0:
                    last = img
                    continue
                diffs.append(sum(cv2.sumElems(cv2.absdiff(last, img))))
                last = img
            new_idx = sorted(range(len(diffs)), key=lambda i: diffs[i], reverse=True)[:keyframes]
        
        self.frame_index = sorted(list(set(self.frame_index).intersection(new_idx)))
        self.tagged_frames = [self.frames[i] for i in self.frame_index]
        self.history.loc[self.history.shape[0]] = [name, thresh, len(self.tagged_frames)]        
        
        self.onsets = [frame_num * (1. / self.fps) for frame_num in self.frame_index]
        
        self.durations = []
        self.elements = []
        for i, f in enumerate(self.frame_index):
            if f != self.frame_index[-1]:
                dur = self.onsets[i+1] - self.onsets[i]
            else:
                dur = (len(self.frames) / self.fps) - self.onsets[i]
            self.durations.append(dur)

            elem = VideoFrameStim(video=self.clip, frame_num=f,
                                  duration=dur)
            self.elements.append(elem)
