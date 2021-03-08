"""
This is my example script
=========================

This example doesn't do much, it just makes a simple plot
"""
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Simple Graph
# ==================
# Example configuring and executing a simple graph. The graph constructed runs on video inputs, and extracts the length of visual text, the amount of vibrance in each frame, and the length of spoken words.

# %%
from pliers.tests.utils import get_test_data_path
from os.path import join
from pliers.stimuli import VideoStim
from pliers.converters import (VideoToAudioConverter,
                               TesseractConverter,
                               WitTranscriptionConverter)
from pliers.extractors import (ExtractorResult,
                               VibranceExtractor,
                               LengthExtractor)
from pliers.graph import Graph


# %%
# Load the stimulus
filename = join(get_test_data_path(), 'video', 'obama_speech.mp4')
video = VideoStim(filename)


# %%
# Configure the graph nodes
nodes = [([(TesseractConverter(), 
              [(LengthExtractor())]), 
            (VibranceExtractor(),)]),
         (VideoToAudioConverter(), 
            [(WitTranscriptionConverter(), 
              [(LengthExtractor())])])]


# %%
# Construct and run the graph
graph = Graph(nodes)
graph.run(video)


# %%
# Save a display of the graph
graph.draw('pliers_simple_graph.png')

# %% [markdown]
# ![title](pliers_simple_graph.png)

# %%



