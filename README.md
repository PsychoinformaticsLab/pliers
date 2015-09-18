# featurex

A Python package for automated extraction of features from multimodal stimuli.

## Installation

> pip install featurex

## Dependencies

* Python packages: numpy, pandas, six (see requirements.txt)
* For image/movie feature extraction, you'll probably need OpenCV (installed with Python bindings)

## Quickstart

In this example, we extract total optical flow in each frame from a short video clip. The DenseOpticalFlowExtractor we'll use relies on the Farneback algorithm implemented in OpenCV. In practice, we might want to experiment a bit with the algorithm's parameters, but for demo purposes, we'll run it with default settings.

```python
from featurex.stims import VideoStim
from featurex.extractors.video import DenseOpticalFlowExtractor

# Load video
video = VideoStim('small.mp4')

# Select extractors to apply
extractors = [DenseOpticalFlowExtractor()]

# Create a timeline by applying all extractors to the movie
timeline = video.extract(extractors)

# Export the timeline to a pandas DataFrame in 'wide' format, where each row is
# a sample and each column is an extracted feature.

print(timeline.to_df(format='long'))

# Outputs something like this...
'''
        onset        name  duration     amplitude
0    0.000000  total_flow  0.033333     22.312111
1    0.033333  total_flow  0.033333   9690.702148
2    0.066667  total_flow  0.033333   7466.022949
3    0.100000  total_flow  0.033333   5140.323730
4    0.133333  total_flow  0.033333   4702.884277
5    0.166667  total_flow  0.033333   2828.656006
6    0.200000  total_flow  0.033333   2349.961914
7    0.233333  total_flow  0.033333   2020.710327
...
'''

```
