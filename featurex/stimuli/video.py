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
        self.elements = [VideoFrameStim(self, i, data=f) for i, f in enumerate(self.frames)]
        self._filter(**kwargs)
        
    def __iter__(self):
        return self.elements.__iter__()

    def filter(self, **kwargs):
        self._filter(**kwargs)

    def _filter(self, every=None, hertz=None, num_frames=None):
        name = "None"
        thresh = 0
        if every is not None:
            name = "every"
            thresh = every
            self.frame_index = range(self.n_frames)[::every]
        elif hertz is not None:
            name = "hertz"
            thresh = hertz
            interval = int(self.fps / hertz)
            self.frame_index = range(self.n_frames)[::interval]
        elif num_frames is not None:
            import cv2
            name = "num_frames"
            thresh = num_frames
            diffs = []
            for i, img in enumerate(self.frames):
                if i == 0:
                    last = img
                    continue
                diffs.append(sum(cv2.sumElems(cv2.absdiff(last, img))))
                last = img
            self.frame_index = sorted(range(len(diffs)), key=lambda i: diffs[i], reverse=True)[:num_frames]
        
        self.tagged_frames = [self.frames[i] for i in self.frame_index]
        self.history.loc[self.history.shape[0]] = [name, thresh, len(self.tagged_frames)]        
        
        onsets = [frame_num * (1. / self.fps) for frame_num in self.frame_index]
        
        self.elements = []
        for i, f in enumerate(self.frame_index):
            if f != self.frame_index[-1]:
                dur = onsets[i+1] - onsets[i]
            else:
                dur = (len(self.frames) / self.fps) - onsets[i]

            elem = VideoFrameStim(video=self.clip, frame_num=f,
                                  duration=dur)
            self.elements.append(elem)
