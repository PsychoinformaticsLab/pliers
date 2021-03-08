"""
This is my example script
=========================

This example doesn't do much, it just makes a simple plot
"""
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Comparing Vision APIs
# ============
# This notebook features the various computer vision APIs that pliers interfaces with. These include the Google Vision, Clarifai, and Indico APIs. To compare their perfomance, image recognition features are extracted from an image of an apple.

# %%
from pliers.tests.utils import get_test_data_path
from os.path import join
from pliers.extractors import (ClarifaiAPIImageExtractor, GoogleVisionAPILabelExtractor)
from pliers.stimuli.image import ImageStim
from pliers.graph import Graph


# %%
# Load the stimulus
stim_path = join(get_test_data_path(), 'image', 'apple.jpg')
stim = ImageStim(stim_path)


# %%
# Configure extractions
clarifai_ext = ClarifaiAPIImageExtractor()
google_ext = GoogleVisionAPILabelExtractor()


# %%
# Run extractions
clarifai_res = clarifai_ext.transform(stim)
indico_res = indico_ext.transform(stim)
google_res = google_ext.transform(stim)


# %%
clarifai_res.to_df()


# %%
df = indico_res.to_df()
df.loc[:, df.sum() > 0.5]


# %%
google_res.to_df()

# %% [markdown]
# Summary
# --------
# For the apple image, it is clear that the Google and Clarifai APIs perform best, as both have "apple", "food", and "fruit" in the top features. On the other hand, the only Indico API feature with a probability over 0.5 is "pomegranate". Furthermore, the Google API seems to also be less noisy than the Clarifai API, where several object labels have probabilities over 0.9.

# %%



