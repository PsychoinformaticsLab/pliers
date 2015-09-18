from featurex import stimuli
from featurex.extractors import Extractor
from featurex.core import Note, Event
import cv2


class ResponseExtractor(Extractor):

    pass


class VideoResponseExtractor(ResponseExtractor):

    target = stimuli.video.VideoStim

    def apply(self, video):

        events = []
        for i, f in enumerate(video):

            img = f.data
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cv2.imshow('frame', img)
            key = cv2.waitKey(100)
            resp = int(key == 32)
            print resp, key

            note = Note(video, self, {'response': resp})
            event = Event(onset=f.onset, duration=f.duration, notes=[note])
            events.append(event)
        return events
