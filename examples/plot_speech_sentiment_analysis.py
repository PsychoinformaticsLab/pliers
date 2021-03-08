"""
This is my example script
=========================

This example doesn't do much, it just makes a simple plot
"""
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Speech sentiment analysis
# ===================
# In this notebook we illustrate the power of pliers converters and extractors in a single pipeline. Specifically, we first run a state-of-the-art speech recognition API to transcribe the text of an audio clip. Then, we run a sentiment analysis API to extract the emotion ratings of the spoken words. The audio clip of this example is a short clip of an Obama administration press conference.
# 
# Note: the analysis is not using any audio features to assist emotion extraction. It is simply only using the text transcribed from the audio

# %%
from pliers.tests.utils import get_test_data_path
from os.path import join
from pliers.stimuli import AudioStim
from pliers.graph import Graph


# %%
# Configure our stimulus and extraction graph
stim = AudioStim(join(get_test_data_path(), 'video', 'obama_speech.mp4'))
nodes = [
    {
        'transformer':'IBMSpeechAPIConverter', 
        'parameters':{'resolution':'phrases'}, 
        'children':[
            {
                'transformer':'IndicoAPITextExtractor',
                'parameters':{'models':['emotion']}
            }
        ]
    }
]
graph = Graph(nodes)

# %% [markdown]
# **Parameters**:
# 
# IBMSpeechAPIConverter - `resolution` specifies how we should chunk the text; using phrases provides better results for emotion analysis, as opposed to word-by-word analysis
# 
# IndicoAPITextExtractor - `models` specifies which analysis models to run using the Indico API; 'emotion' will give back five emotion ratings (anger, joy, fear, sadness, surprise) of the text

# %%
results = graph.run(stim)
results


# %%



