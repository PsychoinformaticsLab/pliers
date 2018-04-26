:tocdepth: 1

Quickstart
==========

The fastest way to learn how pliers works is to work through a few short examples. In this section, we'll demonstrate how pliers can be used to quickly tackle three different feature extraction challenges. We start with very simple examples, and gradually scale up in complexity.

An executable Jupyter Notebook version of this document can be found in the 
`\/examples <https://github.com/tyarkoni/pliers/tree/master/examples>`_ folder of the GitHub repo.

Face detection
--------------
This first example uses the face_recognition package's location extraction method to detect the location of Barack Obama's face within a single image. The tools used to do this are completely local (i.e., the image isn't sent to an external API).

We output the result as a pandas DataFrame; the 'face_locations' column contains the coordinates of the bounding box in CSS format (i.e., top, right, bottom, and left edges).

::

	from pliers.extractors import FaceRecognitionFaceLocationsExtractor

	# A picture of Barack Obama
	image = join(get_test_data_path(), 'image', 'obama.jpg')

	# Initialize Extractor
	ext = FaceRecognitionFaceLocationsExtractor()

	# Apply Extractor to image
	result = ext.transform(image)

	result.to_df()

.. raw:: html

	<style>
		div {
		    overflow-x: auto;
    		max-width: 100%;
		}
	</style>
    <div>
    <style scoped>
    	table {
    		font-size: 12px;
    		margin-bottom: 30px;
    	}
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>onset</th>
          <th>order</th>
          <th>duration</th>
          <th>object_id</th>
          <th>face_locations</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0</td>
          <td>(142, 349, 409, 82)</td>
        </tr>
      </tbody>
    </table>
    </div>



Face detection with multiple inputs
-----------------------------------

What if we want to run the face detector on multiple images? Naively, we
could of course just loop over input images and apply the Extractor to
each one. But pliers makes this even easier for us, by natively
accepting iterables as inputs. The following code is almost identical to
the above snippet. The only notable difference is that, because the
result we get back is now also a list (because the features extracted
from each image are stored separately), we need to explicitly combine
the results using the ``merge_results`` utility.

::

    from pliers.extractors import FaceRecognitionFaceLocationsExtractor, merge_results
    
    images = ['apple.jpg', 'obama.jpg', 'thai_people.jpg']
    images = [join(get_test_data_path(), 'image', img) for img in images]
    
    ext = FaceRecognitionFaceLocationsExtractor()
    results = ext.transform(images)
    df = merge_results(results)
    df

.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>source_file</th>
          <th>onset</th>
          <th>class</th>
          <th>filename</th>
          <th>stim_name</th>
          <th>history</th>
          <th>duration</th>
          <th>order</th>
          <th>object_id</th>
          <th>FaceRecognitionFaceLocationsExtractor#face_locations</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>NaN</td>
          <td>ImageStim</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>obama.jpg</td>
          <td></td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0</td>
          <td>(142, 349, 409, 82)</td>
        </tr>
        <tr>
          <th>1</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>NaN</td>
          <td>ImageStim</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>thai_people.jpg</td>
          <td></td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0</td>
          <td>(236, 862, 325, 772)</td>
        </tr>
        <tr>
          <th>2</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>NaN</td>
          <td>ImageStim</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>thai_people.jpg</td>
          <td></td>
          <td>NaN</td>
          <td>NaN</td>
          <td>1</td>
          <td>(104, 581, 211, 474)</td>
        </tr>
        <tr>
          <th>3</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>NaN</td>
          <td>ImageStim</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>thai_people.jpg</td>
          <td></td>
          <td>NaN</td>
          <td>NaN</td>
          <td>2</td>
          <td>(365, 782, 454, 693)</td>
        </tr>
        <tr>
          <th>4</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>NaN</td>
          <td>ImageStim</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>thai_people.jpg</td>
          <td></td>
          <td>NaN</td>
          <td>NaN</td>
          <td>3</td>
          <td>(265, 444, 355, 354)</td>
        </tr>
      </tbody>
    </table>
    </div>



Note how the merged pandas DataFrame contains 5 rows, even though there
were only 3 input images. The reason is that there are 5 detected faces
across the inputs (0 in the first image, 1 in the second, and 4 in the
third). You can discern the original sources from the ``stim_name`` and
``source_file`` columns.

Face detection using a remote API
---------------------------------

The above examples use an entirely local package (``face_recognition``)
for feature extraction. In this next example, we use the Google Cloud
Vision API to extract various face-related attributes from an image of
Barack Obama. The syntax is identical to the first example, save for the
use of the ``GoogleVisionAPIFaceExtractor`` instead of the
``FaceRecognitionFaceLocationsExtractor``. Note, however, that
successful execution of this code requires you to have a
``GOOGLE_APPLICATION_CREDENTIALS`` environment variable pointing to your
Google credentials JSON file. See the documentation for more details.

::

    from pliers.extractors import GoogleVisionAPIFaceExtractor
    
    ext = GoogleVisionAPIFaceExtractor()
    image = join(get_test_data_path(), 'image', 'obama.jpg')
    result = ext.transform(image)
    
    result.to_df(format='long', timing=False, object_id=False)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>feature</th>
          <th>value</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>face1_boundingPoly_vertex1_x</td>
          <td>34</td>
        </tr>
        <tr>
          <th>1</th>
          <td>face1_boundingPoly_vertex1_y</td>
          <td>3</td>
        </tr>
        <tr>
          <th>2</th>
          <td>face1_boundingPoly_vertex2_x</td>
          <td>413</td>
        </tr>
        <tr>
          <th>3</th>
          <td>face1_boundingPoly_vertex2_y</td>
          <td>3</td>
        </tr>
        <tr>
          <th>4</th>
          <td>face1_boundingPoly_vertex3_x</td>
          <td>413</td>
        </tr>
        <tr>
          <th>5</th>
          <td>face1_boundingPoly_vertex3_y</td>
          <td>444</td>
        </tr>
        <tr>
          <th>6</th>
          <td>face1_boundingPoly_vertex4_x</td>
          <td>34</td>
        </tr>
        <tr>
          <th>7</th>
          <td>face1_boundingPoly_vertex4_y</td>
          <td>444</td>
        </tr>
        <tr>
          <th>8</th>
          <td>face1_fdBoundingPoly_vertex1_x</td>
          <td>81</td>
        </tr>
        <tr>
          <th>9</th>
          <td>face1_fdBoundingPoly_vertex1_y</td>
          <td>112</td>
        </tr>
        <tr>
          <th>10</th>
          <td>face1_fdBoundingPoly_vertex2_x</td>
          <td>367</td>
        </tr>
        <tr>
          <th>11</th>
          <td>face1_fdBoundingPoly_vertex2_y</td>
          <td>112</td>
        </tr>
        <tr>
          <th>12</th>
          <td>face1_fdBoundingPoly_vertex3_x</td>
          <td>367</td>
        </tr>
        <tr>
          <th>13</th>
          <td>face1_fdBoundingPoly_vertex3_y</td>
          <td>397</td>
        </tr>
        <tr>
          <th>14</th>
          <td>face1_fdBoundingPoly_vertex4_x</td>
          <td>81</td>
        </tr>
        <tr>
          <th>15</th>
          <td>face1_fdBoundingPoly_vertex4_y</td>
          <td>397</td>
        </tr>
        <tr>
          <th>16</th>
          <td>face1_landmark_LEFT_EYE_x</td>
          <td>165.82545</td>
        </tr>
        <tr>
          <th>17</th>
          <td>face1_landmark_LEFT_EYE_y</td>
          <td>209.29224</td>
        </tr>
        <tr>
          <th>18</th>
          <td>face1_landmark_LEFT_EYE_z</td>
          <td>-0.0012580488</td>
        </tr>
        <tr>
          <th>19</th>
          <td>face1_landmark_RIGHT_EYE_x</td>
          <td>277.2751</td>
        </tr>
        <tr>
          <th>20</th>
          <td>face1_landmark_RIGHT_EYE_y</td>
          <td>200.76282</td>
        </tr>
        <tr>
          <th>21</th>
          <td>face1_landmark_RIGHT_EYE_z</td>
          <td>-2.2834022</td>
        </tr>
        <tr>
          <th>22</th>
          <td>face1_landmark_LEFT_OF_LEFT_EYEBROW_x</td>
          <td>124.120514</td>
        </tr>
        <tr>
          <th>23</th>
          <td>face1_landmark_LEFT_OF_LEFT_EYEBROW_y</td>
          <td>183.2301</td>
        </tr>
        <tr>
          <th>24</th>
          <td>face1_landmark_LEFT_OF_LEFT_EYEBROW_z</td>
          <td>10.437931</td>
        </tr>
        <tr>
          <th>25</th>
          <td>face1_landmark_RIGHT_OF_LEFT_EYEBROW_x</td>
          <td>191.6638</td>
        </tr>
        <tr>
          <th>26</th>
          <td>face1_landmark_RIGHT_OF_LEFT_EYEBROW_y</td>
          <td>184.7009</td>
        </tr>
        <tr>
          <th>27</th>
          <td>face1_landmark_RIGHT_OF_LEFT_EYEBROW_z</td>
          <td>-23.860262</td>
        </tr>
        <tr>
          <th>28</th>
          <td>face1_landmark_LEFT_OF_RIGHT_EYEBROW_x</td>
          <td>246.78976</td>
        </tr>
        <tr>
          <th>29</th>
          <td>face1_landmark_LEFT_OF_RIGHT_EYEBROW_y</td>
          <td>180.80664</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>100</th>
          <td>face1_landmark_LEFT_EAR_TRAGION_x</td>
          <td>94.670586</td>
        </tr>
        <tr>
          <th>101</th>
          <td>face1_landmark_LEFT_EAR_TRAGION_y</td>
          <td>261.28238</td>
        </tr>
        <tr>
          <th>102</th>
          <td>face1_landmark_LEFT_EAR_TRAGION_z</td>
          <td>144.7621</td>
        </tr>
        <tr>
          <th>103</th>
          <td>face1_landmark_RIGHT_EAR_TRAGION_x</td>
          <td>354.20724</td>
        </tr>
        <tr>
          <th>104</th>
          <td>face1_landmark_RIGHT_EAR_TRAGION_y</td>
          <td>254.42862</td>
        </tr>
        <tr>
          <th>105</th>
          <td>face1_landmark_RIGHT_EAR_TRAGION_z</td>
          <td>139.51318</td>
        </tr>
        <tr>
          <th>106</th>
          <td>face1_landmark_FOREHEAD_GLABELLA_x</td>
          <td>218.83662</td>
        </tr>
        <tr>
          <th>107</th>
          <td>face1_landmark_FOREHEAD_GLABELLA_y</td>
          <td>179.9332</td>
        </tr>
        <tr>
          <th>108</th>
          <td>face1_landmark_FOREHEAD_GLABELLA_z</td>
          <td>-29.149652</td>
        </tr>
        <tr>
          <th>109</th>
          <td>face1_landmark_CHIN_GNATHION_x</td>
          <td>225.09085</td>
        </tr>
        <tr>
          <th>110</th>
          <td>face1_landmark_CHIN_GNATHION_y</td>
          <td>404.05176</td>
        </tr>
        <tr>
          <th>111</th>
          <td>face1_landmark_CHIN_GNATHION_z</td>
          <td>-0.870588</td>
        </tr>
        <tr>
          <th>112</th>
          <td>face1_landmark_CHIN_LEFT_GONION_x</td>
          <td>108.6293</td>
        </tr>
        <tr>
          <th>113</th>
          <td>face1_landmark_CHIN_LEFT_GONION_y</td>
          <td>336.2217</td>
        </tr>
        <tr>
          <th>114</th>
          <td>face1_landmark_CHIN_LEFT_GONION_z</td>
          <td>100.71832</td>
        </tr>
        <tr>
          <th>115</th>
          <td>face1_landmark_CHIN_RIGHT_GONION_x</td>
          <td>342.96274</td>
        </tr>
        <tr>
          <th>116</th>
          <td>face1_landmark_CHIN_RIGHT_GONION_y</td>
          <td>329.56253</td>
        </tr>
        <tr>
          <th>117</th>
          <td>face1_landmark_CHIN_RIGHT_GONION_z</td>
          <td>96.03735</td>
        </tr>
        <tr>
          <th>118</th>
          <td>face1_rollAngle</td>
          <td>-1.6782061</td>
        </tr>
        <tr>
          <th>119</th>
          <td>face1_panAngle</td>
          <td>-1.1388631</td>
        </tr>
        <tr>
          <th>120</th>
          <td>face1_tiltAngle</td>
          <td>-2.0583308</td>
        </tr>
        <tr>
          <th>121</th>
          <td>face1_face_detectionConfidence</td>
          <td>0.999946</td>
        </tr>
        <tr>
          <th>122</th>
          <td>face1_face_landmarkingConfidence</td>
          <td>0.84057003</td>
        </tr>
        <tr>
          <th>123</th>
          <td>face1_joyLikelihood</td>
          <td>VERY_LIKELY</td>
        </tr>
        <tr>
          <th>124</th>
          <td>face1_sorrowLikelihood</td>
          <td>VERY_UNLIKELY</td>
        </tr>
        <tr>
          <th>125</th>
          <td>face1_angerLikelihood</td>
          <td>VERY_UNLIKELY</td>
        </tr>
        <tr>
          <th>126</th>
          <td>face1_surpriseLikelihood</td>
          <td>VERY_UNLIKELY</td>
        </tr>
        <tr>
          <th>127</th>
          <td>face1_underExposedLikelihood</td>
          <td>VERY_UNLIKELY</td>
        </tr>
        <tr>
          <th>128</th>
          <td>face1_blurredLikelihood</td>
          <td>VERY_UNLIKELY</td>
        </tr>
        <tr>
          <th>129</th>
          <td>face1_headwearLikelihood</td>
          <td>VERY_UNLIKELY</td>
        </tr>
      </tbody>
    </table>
    <p>130 rows × 2 columns</p>
    </div>



Notice that the output in this case contains many more features. That’s
because the Google face recognition service gives us back a lot more
information than just the location of the face within the image. Also,
the example illustrates our ability to control the format of the output,
by returning the data in “long” format, and suppressing output of
columns that are uninformative in this context.

Sentiment analysis on text
--------------------------

Here we use the VADER sentiment analyzer (Hutto & Gilbert, 2014)
implemented in the ``nltk`` package to extract sentiment for (a) a
coherent block of text, and (b) each word in the text separately. This
example also introduces the ``Stim`` hierarchy of objects explicitly,
whereas the initialization of ``Stim`` objects was implicit in the
previous examples.

Treat text as a single block
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

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




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>onset</th>
          <th>order</th>
          <th>duration</th>
          <th>object_id</th>
          <th>sentiment_neg</th>
          <th>sentiment_neu</th>
          <th>sentiment_pos</th>
          <th>sentiment_compound</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0</td>
          <td>0.19</td>
          <td>0.51</td>
          <td>0.3</td>
          <td>0.6787</td>
        </tr>
      </tbody>
    </table>
    </div>



Analyze each word individually
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    # Second example: we construct a ComplexTextStim, which will
    # cause each word to be represented as a separate TextStim.
    text = ComplexTextStim(text=raw)
    
    ext = VADERSentimentExtractor()
    results = ext.transform(text)
    
    # Because results is a list of ExtractorResult objects
    # (one per word), we need to merge the results explicitly.
    df = merge_results(results, object_id=False)
    df.head(10)


.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>source_file</th>
          <th>onset</th>
          <th>class</th>
          <th>filename</th>
          <th>stim_name</th>
          <th>history</th>
          <th>duration</th>
          <th>order</th>
          <th>VADERSentimentExtractor#sentiment_compound</th>
          <th>VADERSentimentExtractor#sentiment_neg</th>
          <th>VADERSentimentExtractor#sentiment_neu</th>
          <th>VADERSentimentExtractor#sentiment_pos</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>NaN</td>
          <td>0.0</td>
          <td>TextStim</td>
          <td>NaN</td>
          <td>text[We]</td>
          <td>ComplexTextStim-&gt;ComplexTextIterator/TextStim</td>
          <td>NaN</td>
          <td>0</td>
          <td>0.0000</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>NaN</td>
          <td>0.0</td>
          <td>TextStim</td>
          <td>NaN</td>
          <td>text['re]</td>
          <td>ComplexTextStim-&gt;ComplexTextIterator/TextStim</td>
          <td>NaN</td>
          <td>1</td>
          <td>0.0000</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>NaN</td>
          <td>0.0</td>
          <td>TextStim</td>
          <td>NaN</td>
          <td>text[not]</td>
          <td>ComplexTextStim-&gt;ComplexTextIterator/TextStim</td>
          <td>NaN</td>
          <td>2</td>
          <td>0.0000</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>NaN</td>
          <td>0.0</td>
          <td>TextStim</td>
          <td>NaN</td>
          <td>text[claiming]</td>
          <td>ComplexTextStim-&gt;ComplexTextIterator/TextStim</td>
          <td>NaN</td>
          <td>3</td>
          <td>0.0000</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>NaN</td>
          <td>0.0</td>
          <td>TextStim</td>
          <td>NaN</td>
          <td>text[that]</td>
          <td>ComplexTextStim-&gt;ComplexTextIterator/TextStim</td>
          <td>NaN</td>
          <td>4</td>
          <td>0.0000</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>5</th>
          <td>NaN</td>
          <td>0.0</td>
          <td>TextStim</td>
          <td>NaN</td>
          <td>text[VADER]</td>
          <td>ComplexTextStim-&gt;ComplexTextIterator/TextStim</td>
          <td>NaN</td>
          <td>5</td>
          <td>0.0000</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>6</th>
          <td>NaN</td>
          <td>0.0</td>
          <td>TextStim</td>
          <td>NaN</td>
          <td>text[is]</td>
          <td>ComplexTextStim-&gt;ComplexTextIterator/TextStim</td>
          <td>NaN</td>
          <td>6</td>
          <td>0.0000</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>7</th>
          <td>NaN</td>
          <td>0.0</td>
          <td>TextStim</td>
          <td>NaN</td>
          <td>text[a]</td>
          <td>ComplexTextStim-&gt;ComplexTextIterator/TextStim</td>
          <td>NaN</td>
          <td>7</td>
          <td>0.0000</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>8</th>
          <td>NaN</td>
          <td>0.0</td>
          <td>TextStim</td>
          <td>NaN</td>
          <td>text[very]</td>
          <td>ComplexTextStim-&gt;ComplexTextIterator/TextStim</td>
          <td>NaN</td>
          <td>8</td>
          <td>0.0000</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>9</th>
          <td>NaN</td>
          <td>0.0</td>
          <td>TextStim</td>
          <td>NaN</td>
          <td>text[good]</td>
          <td>ComplexTextStim-&gt;ComplexTextIterator/TextStim</td>
          <td>NaN</td>
          <td>9</td>
          <td>0.4404</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
        </tr>
      </tbody>
    </table>
    </div>



Extract chromagram from an audio clip
-------------------------------------

We have an audio clip, and we’d like to compute its chromagram (i.e., to
extract the normalized energy in each of the 12 pitch classes). This is
trivial thanks to pliers’ support for the ``librosa`` package, which
contains all kinds of useful functions for spectral feature extraction.

::

    from pliers.extractors import ChromaSTFTExtractor
    
    audio = join(get_test_data_path(), 'audio', 'barber.wav')
    # Audio is sampled at 11KHz; let's compute power in 1 sec bins
    ext = ChromaSTFTExtractor(hop_length=11025)
    result = ext.transform(audio).to_df()
    result.head(10)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>onset</th>
          <th>order</th>
          <th>duration</th>
          <th>object_id</th>
          <th>chroma_0</th>
          <th>chroma_1</th>
          <th>chroma_2</th>
          <th>chroma_3</th>
          <th>chroma_4</th>
          <th>chroma_5</th>
          <th>chroma_6</th>
          <th>chroma_7</th>
          <th>chroma_8</th>
          <th>chroma_9</th>
          <th>chroma_10</th>
          <th>chroma_11</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.0</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>0</td>
          <td>0.893229</td>
          <td>0.580649</td>
          <td>0.537203</td>
          <td>0.781329</td>
          <td>0.791074</td>
          <td>0.450180</td>
          <td>0.547222</td>
          <td>0.344074</td>
          <td>0.396035</td>
          <td>0.310631</td>
          <td>0.338300</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.0</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>0</td>
          <td>0.294194</td>
          <td>0.197414</td>
          <td>0.183005</td>
          <td>0.218851</td>
          <td>0.393326</td>
          <td>0.308403</td>
          <td>0.306165</td>
          <td>0.470528</td>
          <td>1.000000</td>
          <td>0.352208</td>
          <td>0.299830</td>
          <td>0.551487</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2.0</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>0</td>
          <td>0.434900</td>
          <td>0.235230</td>
          <td>0.210706</td>
          <td>0.299252</td>
          <td>0.480551</td>
          <td>0.393670</td>
          <td>0.380633</td>
          <td>0.400774</td>
          <td>1.000000</td>
          <td>0.747835</td>
          <td>0.565902</td>
          <td>0.905888</td>
        </tr>
        <tr>
          <th>3</th>
          <td>3.0</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>0</td>
          <td>0.584723</td>
          <td>1.000000</td>
          <td>0.292496</td>
          <td>0.280725</td>
          <td>0.126438</td>
          <td>0.141413</td>
          <td>0.095718</td>
          <td>0.051614</td>
          <td>0.169491</td>
          <td>0.159829</td>
          <td>0.104278</td>
          <td>0.152245</td>
        </tr>
        <tr>
          <th>4</th>
          <td>4.0</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>0</td>
          <td>0.330675</td>
          <td>0.093160</td>
          <td>0.050093</td>
          <td>0.110299</td>
          <td>0.124181</td>
          <td>0.195670</td>
          <td>0.176633</td>
          <td>0.154360</td>
          <td>0.799665</td>
          <td>1.000000</td>
          <td>0.324705</td>
          <td>0.299411</td>
        </tr>
        <tr>
          <th>5</th>
          <td>5.0</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>0</td>
          <td>0.163303</td>
          <td>0.166029</td>
          <td>0.137458</td>
          <td>0.674934</td>
          <td>0.307667</td>
          <td>0.444728</td>
          <td>1.000000</td>
          <td>0.363117</td>
          <td>0.051563</td>
          <td>0.056137</td>
          <td>0.257512</td>
          <td>0.311271</td>
        </tr>
        <tr>
          <th>6</th>
          <td>6.0</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>0</td>
          <td>0.429001</td>
          <td>0.576284</td>
          <td>0.477286</td>
          <td>0.629205</td>
          <td>1.000000</td>
          <td>0.683207</td>
          <td>0.520680</td>
          <td>0.550905</td>
          <td>0.463083</td>
          <td>0.136868</td>
          <td>0.139903</td>
          <td>0.516497</td>
        </tr>
        <tr>
          <th>7</th>
          <td>7.0</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>0</td>
          <td>0.153344</td>
          <td>0.061214</td>
          <td>0.071127</td>
          <td>0.156032</td>
          <td>1.000000</td>
          <td>0.266781</td>
          <td>0.061097</td>
          <td>0.100614</td>
          <td>0.277248</td>
          <td>0.080686</td>
          <td>0.102179</td>
          <td>0.560139</td>
        </tr>
        <tr>
          <th>8</th>
          <td>8.0</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>0</td>
          <td>1.000000</td>
          <td>0.179003</td>
          <td>0.003033</td>
          <td>0.002940</td>
          <td>0.007769</td>
          <td>0.001853</td>
          <td>0.012441</td>
          <td>0.065445</td>
          <td>0.013986</td>
          <td>0.002070</td>
          <td>0.008418</td>
          <td>0.250575</td>
        </tr>
        <tr>
          <th>9</th>
          <td>9.0</td>
          <td>NaN</td>
          <td>1.0</td>
          <td>0</td>
          <td>1.000000</td>
          <td>0.195387</td>
          <td>0.021611</td>
          <td>0.028680</td>
          <td>0.019289</td>
          <td>0.018033</td>
          <td>0.054944</td>
          <td>0.047623</td>
          <td>0.011615</td>
          <td>0.031029</td>
          <td>0.274826</td>
          <td>0.840266</td>
        </tr>
      </tbody>
    </table>
    </div>



::

    # And a plot of the chromagram...
    plt.imshow(result.iloc[:, 4:].values.T, aspect='auto')

.. image:: _static/images/chromagram.png


Sentiment analysis on speech transcribed from audio
---------------------------------------------------

So far all of our examples involve the application of a feature
extractor to an input of the expected modality (e.g., a text sentiment
analyzer applied to text, a face recognizer applied to an image, etc.).
But we often want to extract features that require us to first *convert*
our input to a different modality. Let’s see how pliers handles this
kind of situation.

Say we have an audio clip. We want to run sentiment analysis on the
audio. This requires us to first transcribe any speech contained in the
audio. As it turns out, we don’t have to do anything special here; we
can just feed an audio clip directly to an ``Extractor`` class that
expects a text input (e.g., the ``VADER`` sentiment analyzer we used
earlier). How? Magic! Pliers is smart enough to implicitly convert the
audio clip to a ``ComplexTextStim`` internally. By default, it does this
using IBM’s Watson speech transcription API. Which means you’ll need to
make sure your API key is set up properly in order for the code below to
work. (But if you’d rather use, say, Google’s Cloud Speech API, you
could easily configure pliers to make that the default for audio-to-text
conversion.)

::

    audio = join(get_test_data_path(), 'audio', 'homer.wav')
    ext = VADERSentimentExtractor()
    result = ext.transform(audio)
    df = merge_results(result, object_id=False)
    df

.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>source_file</th>
          <th>onset</th>
          <th>class</th>
          <th>filename</th>
          <th>stim_name</th>
          <th>history</th>
          <th>duration</th>
          <th>order</th>
          <th>VADERSentimentExtractor#sentiment_compound</th>
          <th>VADERSentimentExtractor#sentiment_neg</th>
          <th>VADERSentimentExtractor#sentiment_neu</th>
          <th>VADERSentimentExtractor#sentiment_pos</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>0.04</td>
          <td>TextStim</td>
          <td>NaN</td>
          <td>text[engage]</td>
          <td>AudioStim-&gt;IBMSpeechAPIConverter/ComplexTextSt...</td>
          <td>0.46</td>
          <td>0</td>
          <td>0.34</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>0.50</td>
          <td>TextStim</td>
          <td>NaN</td>
          <td>text[because]</td>
          <td>AudioStim-&gt;IBMSpeechAPIConverter/ComplexTextSt...</td>
          <td>0.37</td>
          <td>1</td>
          <td>0.00</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>0.87</td>
          <td>TextStim</td>
          <td>NaN</td>
          <td>text[we]</td>
          <td>AudioStim-&gt;IBMSpeechAPIConverter/ComplexTextSt...</td>
          <td>0.22</td>
          <td>2</td>
          <td>0.00</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>1.09</td>
          <td>TextStim</td>
          <td>NaN</td>
          <td>text[obey]</td>
          <td>AudioStim-&gt;IBMSpeechAPIConverter/ComplexTextSt...</td>
          <td>0.51</td>
          <td>3</td>
          <td>0.00</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>1.60</td>
          <td>TextStim</td>
          <td>NaN</td>
          <td>text[the]</td>
          <td>AudioStim-&gt;IBMSpeechAPIConverter/ComplexTextSt...</td>
          <td>0.16</td>
          <td>4</td>
          <td>0.00</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>5</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>1.76</td>
          <td>TextStim</td>
          <td>NaN</td>
          <td>text[laws]</td>
          <td>AudioStim-&gt;IBMSpeechAPIConverter/ComplexTextSt...</td>
          <td>0.40</td>
          <td>5</td>
          <td>0.00</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>6</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>2.16</td>
          <td>TextStim</td>
          <td>NaN</td>
          <td>text[of]</td>
          <td>AudioStim-&gt;IBMSpeechAPIConverter/ComplexTextSt...</td>
          <td>0.14</td>
          <td>6</td>
          <td>0.00</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>7</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>2.30</td>
          <td>TextStim</td>
          <td>NaN</td>
          <td>text[thermodynamics]</td>
          <td>AudioStim-&gt;IBMSpeechAPIConverter/ComplexTextSt...</td>
          <td>0.99</td>
          <td>7</td>
          <td>0.00</td>
          <td>0.0</td>
          <td>1.0</td>
          <td>0.0</td>
        </tr>
      </tbody>
    </table>
    </div>



Object recognition on selectively sampled video frames
------------------------------------------------------

A common scenario when analyzing video is to want to apply some kind of
feature extraction tool to individual video frames (i.e., still images).
Often, there’s little to be gained by analyzing every single frame, so
we want to sample frames with some specified frequency. The following
example illustrates how easily this can be accomplished in pliers. It
also demonstrates the concept of *chaining* multiple Transformer
objects. We first convert a video to a series of images, and then apply
an object-detection ``Extractor`` to each image.

Note, as with other examples above, that the ``ClarifaiAPIImageExtractor``
wraps the Clarifai object recognition API, so you’ll need to have an API
key set up appropriately (if you don’t have an API key, and don’t want
to set one up, you can replace ``ClarifaiAPIImageExtractor`` with
``TensorFlowInceptionV3Extractor`` to get similar, though not quite as
accurate, results).

::

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

.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>source_file</th>
          <th>onset</th>
          <th>class</th>
          <th>filename</th>
          <th>stim_name</th>
          <th>history</th>
          <th>duration</th>
          <th>order</th>
          <th>object_id</th>
          <th>ClarifaiAPIImageExtractor#Lego</th>
          <th>...</th>
          <th>ClarifaiAPIImageExtractor#power</th>
          <th>ClarifaiAPIImageExtractor#precision</th>
          <th>ClarifaiAPIImageExtractor#production</th>
          <th>ClarifaiAPIImageExtractor#research</th>
          <th>ClarifaiAPIImageExtractor#robot</th>
          <th>ClarifaiAPIImageExtractor#science</th>
          <th>ClarifaiAPIImageExtractor#still life</th>
          <th>ClarifaiAPIImageExtractor#studio</th>
          <th>ClarifaiAPIImageExtractor#technology</th>
          <th>ClarifaiAPIImageExtractor#toy</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>0.0</td>
          <td>VideoFrameStim</td>
          <td>NaN</td>
          <td>frame[0]</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>0.50</td>
          <td>NaN</td>
          <td>0</td>
          <td>0.949353</td>
          <td>...</td>
          <td>NaN</td>
          <td>0.767964</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.892890</td>
          <td>0.823121</td>
          <td>0.898390</td>
          <td>0.714794</td>
          <td>0.946736</td>
          <td>0.900628</td>
        </tr>
        <tr>
          <th>1</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>0.5</td>
          <td>VideoFrameStim</td>
          <td>NaN</td>
          <td>frame[15]</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>0.50</td>
          <td>NaN</td>
          <td>0</td>
          <td>0.948389</td>
          <td>...</td>
          <td>NaN</td>
          <td>0.743388</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.887668</td>
          <td>0.826262</td>
          <td>0.900226</td>
          <td>0.747545</td>
          <td>0.951705</td>
          <td>0.892195</td>
        </tr>
        <tr>
          <th>2</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>1.0</td>
          <td>VideoFrameStim</td>
          <td>NaN</td>
          <td>frame[30]</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>0.50</td>
          <td>NaN</td>
          <td>0</td>
          <td>0.951566</td>
          <td>...</td>
          <td>NaN</td>
          <td>0.738823</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.885989</td>
          <td>0.801925</td>
          <td>0.908438</td>
          <td>0.756304</td>
          <td>0.948202</td>
          <td>0.903330</td>
        </tr>
        <tr>
          <th>3</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>1.5</td>
          <td>VideoFrameStim</td>
          <td>NaN</td>
          <td>frame[45]</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>0.50</td>
          <td>NaN</td>
          <td>0</td>
          <td>0.951050</td>
          <td>...</td>
          <td>NaN</td>
          <td>0.794678</td>
          <td>0.710889</td>
          <td>0.749307</td>
          <td>0.893252</td>
          <td>0.892987</td>
          <td>0.877005</td>
          <td>NaN</td>
          <td>0.962567</td>
          <td>0.857956</td>
        </tr>
        <tr>
          <th>4</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>2.0</td>
          <td>VideoFrameStim</td>
          <td>NaN</td>
          <td>frame[60]</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>0.50</td>
          <td>NaN</td>
          <td>0</td>
          <td>0.872721</td>
          <td>...</td>
          <td>0.756543</td>
          <td>0.802734</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.866742</td>
          <td>0.816107</td>
          <td>0.802523</td>
          <td>NaN</td>
          <td>0.956920</td>
          <td>0.803250</td>
        </tr>
        <tr>
          <th>5</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>2.5</td>
          <td>VideoFrameStim</td>
          <td>NaN</td>
          <td>frame[75]</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>0.50</td>
          <td>NaN</td>
          <td>0</td>
          <td>0.930966</td>
          <td>...</td>
          <td>NaN</td>
          <td>0.763779</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.841595</td>
          <td>0.755196</td>
          <td>0.885707</td>
          <td>0.713024</td>
          <td>0.937848</td>
          <td>0.876500</td>
        </tr>
        <tr>
          <th>6</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>3.0</td>
          <td>VideoFrameStim</td>
          <td>NaN</td>
          <td>frame[90]</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>0.50</td>
          <td>NaN</td>
          <td>0</td>
          <td>0.866936</td>
          <td>...</td>
          <td>0.749151</td>
          <td>0.749939</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.862391</td>
          <td>0.824693</td>
          <td>0.806569</td>
          <td>NaN</td>
          <td>0.948547</td>
          <td>0.793848</td>
        </tr>
        <tr>
          <th>7</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>3.5</td>
          <td>VideoFrameStim</td>
          <td>NaN</td>
          <td>frame[105]</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>0.50</td>
          <td>NaN</td>
          <td>0</td>
          <td>0.957496</td>
          <td>...</td>
          <td>NaN</td>
          <td>0.775053</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.895434</td>
          <td>0.839599</td>
          <td>0.890773</td>
          <td>0.720677</td>
          <td>0.949031</td>
          <td>0.898136</td>
        </tr>
        <tr>
          <th>8</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>4.0</td>
          <td>VideoFrameStim</td>
          <td>NaN</td>
          <td>frame[120]</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>0.50</td>
          <td>NaN</td>
          <td>0</td>
          <td>0.954910</td>
          <td>...</td>
          <td>NaN</td>
          <td>0.785069</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.888534</td>
          <td>0.833464</td>
          <td>0.895954</td>
          <td>0.752757</td>
          <td>0.948506</td>
          <td>0.897712</td>
        </tr>
        <tr>
          <th>9</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>4.5</td>
          <td>VideoFrameStim</td>
          <td>NaN</td>
          <td>frame[135]</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>0.50</td>
          <td>NaN</td>
          <td>0</td>
          <td>0.957653</td>
          <td>...</td>
          <td>NaN</td>
          <td>0.796410</td>
          <td>0.711184</td>
          <td>NaN</td>
          <td>0.897311</td>
          <td>0.854389</td>
          <td>0.899367</td>
          <td>0.726466</td>
          <td>0.951222</td>
          <td>0.893269</td>
        </tr>
        <tr>
          <th>10</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>5.0</td>
          <td>VideoFrameStim</td>
          <td>NaN</td>
          <td>frame[150]</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>0.50</td>
          <td>NaN</td>
          <td>0</td>
          <td>0.954066</td>
          <td>...</td>
          <td>NaN</td>
          <td>0.793047</td>
          <td>0.717981</td>
          <td>NaN</td>
          <td>0.904960</td>
          <td>0.861293</td>
          <td>0.905260</td>
          <td>0.754906</td>
          <td>0.956006</td>
          <td>0.894970</td>
        </tr>
        <tr>
          <th>11</th>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
          <td>5.5</td>
          <td>VideoFrameStim</td>
          <td>NaN</td>
          <td>frame[165]</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>0.07</td>
          <td>NaN</td>
          <td>0</td>
          <td>0.932649</td>
          <td>...</td>
          <td>NaN</td>
          <td>0.818984</td>
          <td>0.758780</td>
          <td>NaN</td>
          <td>0.876721</td>
          <td>0.882386</td>
          <td>0.887411</td>
          <td>NaN</td>
          <td>0.958058</td>
          <td>0.872935</td>
        </tr>
      </tbody>
    </table>
    <p>12 rows × 41 columns</p>
    </div>



The resulting data frame has 41 columns (!), most of which are
individual object labels like ‘lego’, ‘toy’, etc., selected for us by
the Clarifai API on the basis of the content detected in the video (we
could have also forced the API to return values for specific labels).

Multiple extractors
-------------------

So far we’ve only used a single ``Extractor`` at a time to extract
information from our inputs. Now we’ll start to get a little more
ambitious. Let’s say we have a video that we want to extract *lots* of
different features from–in multiple modalities. Specifically, we want to
extract all of the following:

-  Object recognition and face detection applied to every 10th frame of
   the video;
-  A second-by-second estimate of spectral power in the speech frequency
   band;
-  A word-by-word speech transcript;
-  Estimates of several lexical properties (e.g., word length, written
   word frequency, etc.) for every word in the transcript;
-  Sentiment analysis applied to the entire transcript.

We’ve already seen some of these features extracted individually, but
now we’re going to extract *all* of them at once. As it turns out, the
code looks almost exactly like a concatenated version of several of our
examples above.

::

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


.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>object_id</th>
          <th>onset</th>
          <th>order</th>
          <th>duration</th>
          <th>feature</th>
          <th>value</th>
          <th>extractor</th>
          <th>stim_name</th>
          <th>class</th>
          <th>filename</th>
          <th>history</th>
          <th>source_file</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2</th>
          <td>0</td>
          <td>0.000000</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.970786</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[0]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
        <tr>
          <th>296</th>
          <td>0</td>
          <td>0.833333</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.976996</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[10]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
        <tr>
          <th>592</th>
          <td>0</td>
          <td>1.666667</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.972223</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[20]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
        <tr>
          <th>887</th>
          <td>0</td>
          <td>2.500000</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.98288</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[30]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
        <tr>
          <th>1198</th>
          <td>0</td>
          <td>3.333333</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.94764</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[40]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
        <tr>
          <th>1492</th>
          <td>0</td>
          <td>4.166667</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.952409</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[50]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
        <tr>
          <th>1795</th>
          <td>0</td>
          <td>5.000000</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.951445</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[60]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
        <tr>
          <th>2096</th>
          <td>0</td>
          <td>5.833333</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.954552</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[70]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
        <tr>
          <th>2392</th>
          <td>0</td>
          <td>6.666667</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.953084</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[80]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
        <tr>
          <th>2695</th>
          <td>0</td>
          <td>7.500000</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.947371</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[90]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
      </tbody>
    </table>
    </div>



The resulting pandas DataFrame is quite large; even for our 9-second
video, we get back over 3,000 rows! Importantly, though, the DataFrame
contains all kinds of metadata that makes it easy to filter and sort the
results in whatever way we might want to (e.g., we can filter on the
extractor, stim class, onset or duration, etc.).

Multiple extractors with a Graph
--------------------------------

The above code listing is already pretty terse, and has the advantage of
being explicit about every step. But if it’s brevity we’re after, pliers
is happy to oblige us. The package includes a ``Graph`` abstraction that
allows us to load an arbitrary number of ``Transformer`` into a graph,
and execute them all in one shot. The code below is functionally
identical to the last example, but only about the third of the length.
It also requires fewer imports, since ``Transformer`` objects that we
don’t need to initialize with custom arguments can be passed to the
``Graph`` as strings.

The upshot of all this is that, in just a few lines of Python code,
we’re abvle to extract a broad range of multimodal features from video,
image, audio or text inputs, using state-of-the-art tools and services!

::

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


.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>object_id</th>
          <th>onset</th>
          <th>order</th>
          <th>duration</th>
          <th>feature</th>
          <th>value</th>
          <th>extractor</th>
          <th>stim_name</th>
          <th>class</th>
          <th>filename</th>
          <th>history</th>
          <th>source_file</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2</th>
          <td>0</td>
          <td>0.000000</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.970786</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[0]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
        <tr>
          <th>296</th>
          <td>0</td>
          <td>0.833333</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.976996</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[10]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
        <tr>
          <th>592</th>
          <td>0</td>
          <td>1.666667</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.972223</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[20]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
        <tr>
          <th>887</th>
          <td>0</td>
          <td>2.500000</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.98288</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[30]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
        <tr>
          <th>1198</th>
          <td>0</td>
          <td>3.333333</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.94764</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[40]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
        <tr>
          <th>1492</th>
          <td>0</td>
          <td>4.166667</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.952409</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[50]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
        <tr>
          <th>1795</th>
          <td>0</td>
          <td>5.000000</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.951445</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[60]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
        <tr>
          <th>2096</th>
          <td>0</td>
          <td>5.833333</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.954552</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[70]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
        <tr>
          <th>2392</th>
          <td>0</td>
          <td>6.666667</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.953084</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[80]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
        <tr>
          <th>2695</th>
          <td>0</td>
          <td>7.500000</td>
          <td>NaN</td>
          <td>0.833333</td>
          <td>administration</td>
          <td>0.947371</td>
          <td>ClarifaiAPIImageExtractor</td>
          <td>frame[90]</td>
          <td>VideoFrameStim</td>
          <td>None</td>
          <td>VideoStim-&gt;FrameSamplingFilter/VideoFrameColle...</td>
          <td>/Users/tal/Dropbox/Code/pliers/pliers/tests/da...</td>
        </tr>
      </tbody>
    </table>
    </div>