# Annotations

A Python package for (mostly) automated stimulus annotation.

## Installation

> pip install annotations

## Dependencies

* Python packages: numpy, pandas, six (see requirements.txt)
* For image/movie annotation, you'll probably OpenCV (installed with Python bindings)

## Quickstart

In this example, we extract total optical flow in each frame from a short video clip. The DenseOpticalFlowAnnotator we'll use relies on the Farneback algorithm implemented in OpenCV. In practice, we might want to experiment a bit with the algorithm's parameters, but for demo purposes, we'll run it with default settings.

```python
from annotations.stims import VideoStim
from annotations.annotators.video import DenseOpticalFlowAnnotator

# Load video
video = VideoStim('small.mp4')

# Select annotators to apply
annotators = [DenseOpticalFlowAnnotator()]

# Create a timeline by applying all annotators to the movie
timeline = video.annotate(annotators)

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
