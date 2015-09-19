from featurex import stimuli
from featurex.extractors import StimExtractor
from featurex.core import Value, Event
import cv2


class ResponseExtractor(StimExtractor):

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

            value = Value(video, self, {'response': resp})
            event = Event(onset=f.onset, duration=f.duration, values=[value])
            events.append(event)
        return events
