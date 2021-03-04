"""
This is my example script
=========================

This example doesn't do much, it just makes a simple plot
"""
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
# Example-specific imports are in individual cells below; here we
# just import stuff we reuse repeatedly.
from pliers.extractors import merge_results
from pliers.tests.utils import get_test_data_path
from os.path import join
from matplotlib import pyplot as plt

# %% [markdown]
# # Pliers Quickstart
# This notebook contains a few examples that demonstrate how to extract various kinds of features with pliers. We start with very simple examples, and gradually scale up in complexity.
# 
# ## Face detection
# This first example uses the face_recognition package's location extraction method to detect the location of Barack Obama's face within a single image. The tools used to do this are completely local (i.e., the image isn't sent to an external API).
# 
# We output the result as a pandas DataFrame; the `'face_locations`' column contains the coordinates of the bounding box in CSS format (i.e., top, right, bottom, and left edges).

# %%
from pliers.extractors import FaceRecognitionFaceLocationsExtractor

# A picture of Barack Obama
image = join(get_test_data_path(), 'image', 'obama.jpg')

# Initialize Extractor
ext = FaceRecognitionFaceLocationsExtractor()

# Apply Extractor to image
result = ext.transform(image)

result.to_df()

# %% [markdown]
# ## Face detection with multiple inputs
# What if we want to run the face detector on multiple images? Naively, we could of course just loop over input images and apply the Extractor to each one. But pliers makes this even easier for us, by natively accepting iterables as inputs. The following code is almost identical to the above snippet. The only notable difference is that, because the result we get back is now also a list (because the features extracted from each image are stored separately), we need to explicitly combine the results using the `merge_results` utility.

# %%
from pliers.extractors import FaceRecognitionFaceLocationsExtractor, merge_results

images = ['apple.jpg', 'obama.jpg', 'thai_people.jpg']
images = [join(get_test_data_path(), 'image', img) for img in images]

ext = FaceRecognitionFaceLocationsExtractor()
results = ext.transform(images)
df = merge_results(results)
df

# %% [markdown]
# Note how the merged pandas DataFrame contains 5 rows, even though there were only 3 input images. The reason is that there are 5 detected faces across the inputs (0 in the first image, 1 in the second, and 4 in the third). You can discern the original sources from the `stim_name` and `source_file` columns.
# 
# ## Face detection using a remote API
# The above examples use an entirely local package (`face_recognition`) for feature extraction. In this next example, we use the Google Cloud Vision API to extract various face-related attributes from an image of Barack Obama. The syntax is identical to the first example, save for the use of the `GoogleVisionAPIFaceExtractor` instead of the `FaceRecognitionFaceLocationsExtractor`. Note, however, that successful execution of this code requires you to have a `GOOGLE_APPLICATION_CREDENTIALS` environment variable pointing to your Google credentials JSON file. See the documentation for more details.

# %%
from pliers.extractors import GoogleVisionAPIFaceExtractor

ext = GoogleVisionAPIFaceExtractor()
image = join(get_test_data_path(), 'image', 'obama.jpg')
result = ext.transform(image)

result.to_df(format='long', timing=False, object_id=False)

# %% [markdown]
# Notice that the output in this case contains many more features. That's because the Google face recognition service gives us back a lot more information than just the location of the face within the image. Also, the example illustrates our ability to control the format of the output, by returning the data in "long" format, and suppressing output of columns that are uninformative in this context.
# %% [markdown]
# ## Sentiment analysis on text
# Here we use the VADER sentiment analyzer (Hutto & Gilbert, 2014) implemented in the `nltk` package to extract sentiment for (a) a coherent block of text, and (b) each word in the text separately. This example also introduces the `Stim` hierarchy of objects explicitly, whereas the initialization of `Stim` objects was implicit in the previous examples.
# 
# #### Treat text as a single block

# %%
from pliers.stimuli import TextStim, ComplexTextStim
from pliers.extractors import VADERSentimentExtractor, merge_results

raw = """We're not claiming that VADER is a very good sentiment analysis tool.
Sentiment analysis is a really, really difficult problem. But just to make a
point, here are some clearly valenced words: disgusting, wonderful, poop,
sunshine, smile."""

# First example: we treat all text as part of a single token
text = TextStim(text=raw)

ext = VADERSentimentExtractor()
results = ext.transform(text)
results.to_df()

# %% [markdown]
# #### Analyze each word individually

# %%
# Second example: we construct a ComplexTextStim, which will
# cause each word to be represented as a separate TextStim.
text = ComplexTextStim(text=raw)

ext = VADERSentimentExtractor()
results = ext.transform(text)

# Because results is a list of ExtractorResult objects
# (one per word), we need to merge the results explicitly.
df = merge_results(results, object_id=False)
df.head(10)

# %% [markdown]
# ## Extract chromagram from an audio clip
# We have an audio clip, and we'd like to compute its chromagram (i.e., to extract the normalized energy in each of the 12 pitch classes). This is trivial thanks to pliers' support for the `librosa` package, which contains all kinds of useful functions for spectral feature extraction.

# %%
from pliers.extractors import ChromaSTFTExtractor

audio = join(get_test_data_path(), 'audio', 'barber.wav')
# Audio is sampled at 11KHz; let's compute power in 1 sec bins
ext = ChromaSTFTExtractor(hop_length=11025)
result = ext.transform(audio).to_df()
result.head(10)


# %%
# And a plot of the chromagram...
plt.imshow(result.iloc[:, 4:].values.T, aspect='auto')

# %% [markdown]
# ## Sentiment analysis on speech transcribed from audio
# So far all of our examples involve the application of a feature extractor to an input of the expected modality (e.g., a text sentiment analyzer applied to text, a face recognizer applied to an image, etc.). But we often want to extract features that require us to first *convert* our input to a different modality. Let's see how pliers handles this kind of situation.
# 
# Say we have an audio clip. We want to run sentiment analysis on the audio. This requires us to first transcribe any speech contained in the audio. As it turns out, we don't have to do anything special here; we can just feed an audio clip directly to an `Extractor` class that expects a text input (e.g., the `VADER` sentiment analyzer we used earlier). How? Magic! Pliers is smart enough to implicitly convert the audio clip to a `ComplexTextStim` internally. By default, it does this using IBM's Watson speech transcription API. Which means you'll need to make sure your API key is set up properly in order for the code below to work. (But if you'd rather use, say, Google's Cloud Speech API, you could easily configure pliers to make that the default for audio-to-text conversion.)

# %%
audio = join(get_test_data_path(), 'audio', 'homer.wav')
ext = VADERSentimentExtractor()
result = ext.transform(audio)
df = merge_results(result, object_id=False)
df

# %% [markdown]
# ## Object recognition on selectively sampled video frames
# A common scenario when analyzing video is to want to apply some kind of feature extraction tool to individual video frames (i.e., still images). Often, there's little to be gained by analyzing every single frame, so we want to sample frames with some specified frequency. The following example illustrates how easily this can be accomplished in pliers. It also demonstrates the concept of *chaining* multiple Transformer objects. We first convert a video to a series of images, and then apply an object-detection `Extractor` to each image.
# 
# Note, as with other examples above, that the `ClarifaiAPIImageExtractor` wraps the Clarifai object recognition API, so you'll need to have an API key set up appropriately (if you don't have an API key, and don't want to set one up, you can replace `ClarifaiAPIExtractor` with `TensorFlowInceptionV3Extractor` to get similar, though not quite as accurate, results).

# %%
from pliers.filters import FrameSamplingFilter
from pliers.extractors import ClarifaiAPIImageExtractor, merge_results

video = join(get_test_data_path(), 'video', 'small.mp4')

# Sample 2 frames per second
sampler = FrameSamplingFilter(hertz=2)
frames = sampler.transform(video)

ext = ClarifaiAPIImageExtractor()
results = ext.transform(frames)
df = merge_results(results, )
df

# %% [markdown]
# The resulting data frame has 41 columns (!), most of which are individual object labels like 'lego', 'toy', etc., selected for us by the Clarifai API on the basis of the content detected in the video (we could have also forced the API to return values for specific labels).
# %% [markdown]
# ## Multiple extractors
# So far we've only used a single `Extractor` at a time to extract information from our inputs. Now we'll start to get a little more ambitious. Let's say we have a video that we want to extract *lots* of different features from--in multiple modalities. Specifically, we want to extract all of the following:
# 
# * Object recognition and face detection applied to every 10th frame of the video;
# * A second-by-second estimate of spectral power in the speech frequency band;
# * A word-by-word speech transcript;
# * Estimates of several lexical properties (e.g., word length, written word frequency, etc.) for every word in the transcript;
# * Sentiment analysis applied to the entire transcript.
# 
# We've already seen some of these features extracted individually, but now we're going to extract *all* of them at once. As it turns out, the code looks almost exactly like a concatenated version of several of our examples above.

# %%
from pliers.tests.utils import get_test_data_path
from os.path import join
from pliers.filters import FrameSamplingFilter
from pliers.converters import GoogleSpeechAPIConverter
from pliers.extractors import (ClarifaiAPIImageExtractor, GoogleVisionAPIFaceExtractor,
                               ComplexTextExtractor, PredefinedDictionaryExtractor,
                               STFTAudioExtractor, VADERSentimentExtractor,
                               merge_results)

video = join(get_test_data_path(), 'video', 'obama_speech.mp4')

# Store all the returned features in a single list (nested lists
# are fine, the merge_results function will flatten everything)
features = []

# Sample video frames and apply the image-based extractors
sampler = FrameSamplingFilter(every=10)
frames = sampler.transform(video)

obj_ext = ClarifaiAPIImageExtractor()
obj_features = obj_ext.transform(frames)
features.append(obj_features)

face_ext = GoogleVisionAPIFaceExtractor()
face_features = face_ext.transform(frames)
features.append(face_features)

# Power in speech frequencies
stft_ext = STFTAudioExtractor(freq_bins=[(100, 300)])
speech_features = stft_ext.transform(video)
features.append(speech_features)

# Explicitly transcribe the video--we could also skip this step
# and it would be done implicitly, but this way we can specify
# that we want to use the Google Cloud Speech API rather than
# the package default (IBM Watson)
text_conv = GoogleSpeechAPIConverter()
text = text_conv.transform(video)
                  
# Text-based features
text_ext = ComplexTextExtractor()
text_features = text_ext.transform(text)
features.append(text_features)

dict_ext = PredefinedDictionaryExtractor(
    variables=['affect/V.Mean.Sum', 'subtlexusfrequency/Lg10WF'])
norm_features = dict_ext.transform(text)
features.append(norm_features)

sent_ext = VADERSentimentExtractor()
sent_features = sent_ext.transform(text)
features.append(sent_features)

# Ask for data in 'long' format, and code extractor name as a separate
# column instead of prepending it to feature names.
df = merge_results(features, format='long', extractor_names='column')

# Output rows in a sensible order
df.sort_values(['extractor', 'feature', 'onset', 'duration', 'order']).head(10)

# %% [markdown]
# The resulting pandas DataFrame is quite large; even for our 9-second video, we get back over 3,000 rows! Importantly, though, the DataFrame contains all kinds of metadata that makes it easy to filter and sort the results in whatever way we might want to (e.g., we can filter on the extractor, stim class, onset or duration, etc.).
# %% [markdown]
# ## Multiple extractors with a Graph
# The above code listing is already pretty terse, and has the advantage of being explicit about every step. But if it's brevity we're after, pliers is happy to oblige us. The package includes a `Graph` abstraction that allows us to load an arbitrary number of `Transformer` into a graph, and execute them all in one shot. The code below is functionally identical to the last example, but only about the third of the length. It also requires fewer imports, since `Transformer` objects that we don't need to initialize with custom arguments can be passed to the `Graph` as strings. 
# 
# The upshot of all this is that, in just a few lines of Python code, we're abvle to extract a broad range of multimodal features from video, image, audio or text inputs, using state-of-the-art tools and services!

# %%
from pliers.tests.utils import get_test_data_path
from os.path import join
from pliers.graph import Graph
from pliers.filters import FrameSamplingFilter
from pliers.extractors import (PredefinedDictionaryExtractor, STFTAudioExtractor,
                               merge_results)


video = join(get_test_data_path(), 'video', 'obama_speech.mp4')

# Define nodes
nodes = [
    (FrameSamplingFilter(every=10),
         ['ClarifaiAPIImageExtractor', 'GoogleVisionAPIFaceExtractor']),
    (STFTAudioExtractor(freq_bins=[(100, 300)])),
    ('GoogleSpeechAPIConverter',
         ['ComplexTextExtractor',
          PredefinedDictionaryExtractor(['affect/V.Mean.Sum',
                                         'subtlexusfrequency/Lg10WF']),
         'VADERSentimentExtractor'])
]

# Initialize and execute Graph
g = Graph(nodes)

# Arguments to merge_results can be passed in here
df = g.transform(video, format='long', extractor_names='column')

# Output rows in a sensible order
df.sort_values(['extractor', 'feature', 'onset', 'duration', 'order']).head(10)


