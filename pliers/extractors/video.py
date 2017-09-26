'''
Extractors that operate primarily or exclusively on Video stimuli.
'''

from pliers.stimuli.video import VideoStim
from pliers.extractors.base import Extractor, ExtractorResult
from pliers.utils import attempt_to_import, verify_dependencies

import numpy as np

cv2 = attempt_to_import('cv2')


class VideoExtractor(Extractor):

    ''' Base Video Extractor class; all subclasses can only be applied to
    video. '''
    _input_type = VideoStim


class FarnebackOpticalFlowExtractor(VideoExtractor):

    ''' Extracts total amount of dense optical flow between every pair of video
    frames.

    Args:
        pyr_scale (float): specifying the image scale (<1) to build pyramids
            for each image; pyr_scale=0.5 means a classical pyramid, where
            each next layer is twice smaller than the previous one.
        levels (int): number of pyramid layers including the initial image;
            levels=1 means that no extra layers are created and only the
            original images are used.
        winsize (int): averaging window size; larger values increase the
            algorithm robustness to image noise and give more chances for fast
            motion detection, but yield more blurred motion field.
        iterations (int): number of iterations the algorithm does at each
            pyramid level
        poly_n (int): size of the pixel neighborhood used to find polynomial
            expansion in each pixel; larger values mean that the image will be
            approximated with smoother surfaces, yielding more robust algorithm
            and more blurred motion field, typically poly_n =5 or 7.
        poly_sigma (float):  standard deviation of the Gaussian that is used to
            smooth derivatives used as a basis for the polynomial expansion;
            for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good
            value would be poly_sigma=1.5.
        flags (int): operation flags, usually OPTFLOW_USE_INITIAL_FLOW or
            OPTFLOW_FARNEBACK_GAUSSIAN
        show (bool): flag for displaying flow image during extraction
    '''

    _log_attributes = ('pyr_scale', 'levels', 'winsize', 'iterations',
                       'poly_n', 'poly_sigma', 'flags')

    def __init__(self, pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                 poly_n=5, poly_sigma=1.2, flags=0, show=False):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags
        self.show = show
        super(FarnebackOpticalFlowExtractor, self).__init__()

    def _extract(self, stim):
        verify_dependencies(['cv2'])
        flows = []
        onsets = []
        durations = []
        for i, f in enumerate(stim):

            frame = f.data
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if i == 0:
                last_frame = frame

            flow = cv2.calcOpticalFlowFarneback(last_frame, frame, None,
                                                self.pyr_scale, self.levels,
                                                self.winsize, self.iterations,
                                                self.poly_n, self.poly_sigma,
                                                self.flags)
            flow = np.sqrt((flow ** 2).sum(2))

            if self.show:
                cv2.imshow('frame', flow.astype('int8'))
                cv2.waitKey(1)

            last_frame = frame
            flows.append(flow.sum())
            onsets.append(f.onset)
            durations.append(f.duration)

        return ExtractorResult(flows, stim, self, features=['total_flow'],
                               onsets=onsets, durations=durations)
