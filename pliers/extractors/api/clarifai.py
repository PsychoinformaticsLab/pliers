'''
Extractors that interact with the Clarifai API.
'''

import logging
import os
from contextlib import ExitStack

import pandas as pd

from pliers.extractors.image import ImageExtractor
from pliers.extractors.video import VideoExtractor
from pliers.extractors.base import ExtractorResult
from pliers.transformers import BatchTransformerMixin
from pliers.transformers.api import APITransformer
from pliers.utils import listify, attempt_to_import, verify_dependencies


clarifai_channel = attempt_to_import('clarifai_grpc.channel.clarifai_channel', 'clarifai_channel',
                                    ['ClarifaiChannel'])

clarifai_api = attempt_to_import('clarifai_grpc.grpc.api', 'clarifai_api', ['resources_pb2, service_pb2, service_pb2_grpc'])

class ClarifaiAPIExtractor(APITransformer):

    ''' Uses the Clarifai API to extract tags of visual stimuli.

    Args:
        api_key (str): A valid API_KEY for the Clarifai API. Only needs to be
            passed the first time the extractor is initialized.
        model (str): The name of the Clarifai model to use. If None, defaults
            to the general image tagger.
        min_value (float): A value between 0.0 and 1.0 indicating the minimum
            confidence required to return a prediction. Defaults to 0.0.
        max_concepts (int): A value between 0 and 200 indicating the maximum
            number of label predictions returned.
        select_concepts (list): List of concepts (strings) to query from the
            API. For example, ['food', 'animal'].
        rate_limit (int): The minimum number of seconds required between
            transform calls on this Transformer.
    '''

    _log_attributes = ('access_token', 'model_name', 'min_value',
                       'max_concepts', 'select_concepts')
    _env_keys = ('CLARIFAI_ACCESS_TOKEN', 'CLARIFAI_APP_ID', 'CLARIFAI_USER_ID')
    VERSION = '1.0'

    def __init__(self, access_token=None, user_id=None, app_id=None, model='general-image-recognition', min_value=None,
                 max_concepts=None, select_concepts=None, rate_limit=None,
                 batch_size=None):
        verify_dependencies(['clarifai_channel', 'clarifai_api'])
        if access_token is None:
            try:
                access_token = os.environ['CLARIFAI_ACCESS_TOKEN']
            except KeyError:
                raise ValueError("A valid Clarifai API ACCESS_TOKEN "
                                 "must be passed the first time a Clarifai "
                                 "extractor is initialized.")

        if user_id is None:
            try:
                user_id = os.environ['CLARIFAI_USER_ID']
            except KeyError:
                raise ValueError("A valid Clarifai API CLARIFAI_USER_ID "
                                 "must be passed the first time a Clarifai "
                                 "extractor is initialized.")

        if app_id is None:
            try:
                app_id = os.environ['CLARIFAI_APP_ID']
            except KeyError:
                raise ValueError("A valid Clarifai API CLARIFAI_APP_ID "
                                 "must be passed the first time a Clarifai "
                                 "extractor is initialized.")

        self.access_token = access_token
        self.api = clarifai_api.service_pb2_grpc.V2Stub(clarifai_channel.ClarifaiChannel.get_grpc_channel())
        self.metadata =  (('authorization', 'Key ' + access_token),)
        self.user_id= user_id
        self.app_id = app_id

        self.model_name = model
        self.application_id = None
        self.min_value = min_value #NA 
        self.max_concepts = max_concepts
        self.select_concepts = select_concepts
        if select_concepts:
            select_concepts = listify(select_concepts)
            self.select_concepts = [clarifai_api.resources_pb2.Concept(name=n)
                                    for n in select_concepts]
        super().__init__(rate_limit=rate_limit)

    @property
    def api_keys(self):
        return [self.access_token]

    def check_valid_keys(self):
        return None

    def _query_api(self, objects):
        verify_dependencies(['clarifai_api'])
        model_options = None
        if self.select_concepts or self.max_concepts or self.min_value:
            model_options = clarifai_api.resources_pb2.Model(
            output_info=clarifai_api.resources_pb2.OutputInfo(
                output_config=clarifai_api.resources_pb2.OutputConfig(
                    select_concepts=self.select_concepts,
                    min_value = self.min_value,
                    max_concepts = self.max_concepts
                )
            )
        )
        request = clarifai_api.service_pb2.PostModelOutputsRequest(
            model_id=self.model_name,
            user_app_id=clarifai_api.resources_pb2.UserAppIDSet(user_id=self.user_id, app_id=self.app_id),
            inputs=objects,
            model=model_options
        )
        response = self.api.PostModelOutputs(request, metadata=self.metadata)
        return response.outputs

    def _parse_annotations(self, annotation, handle_annotations=None):
        """
        Parse outputs from a clarifai extraction.

        Args:
            handle_annotations (str): How returned face annotations should be
                handled in cases where there are multiple faces.
                'first' indicates to only use the first face JSON object, all
                other values will default to including every face.
        """
        # check whether the model is the face detection model
        if self.model_name == 'face-detection':

            # if a face was detected, get at least the boundaries
            if annotation.data:
                # if specified, only return first face
                if handle_annotations == 'first':
                    annotation = [annotation.data.regions[0]]
                # else collate all faces into a multi-row dataframe
                face_results = []
                for i, d in enumerate(annotation.data.regions):
                    data_dict = {}
                    for k, v in d.region_info.bounding_box.ListFields():
                        data_dict[k.name] = v

                    for tag in d.data.concepts:
                        data_dict[tag.name] = tag.value
                    
                    face_results.append(data_dict)
                return face_results
            # return an empty dict if there was no face
        else:
            data_dict = {}
            for tag in annotation.data.concepts:
                data_dict[tag.name] = tag.value
            return data_dict


class ClarifaiAPIImageExtractor(ClarifaiAPIExtractor, BatchTransformerMixin,
                                ImageExtractor):

    ''' Uses the Clarifai API to extract tags of images.

    Args:
        api_key (str): A valid API_KEY for the Clarifai API. Only needs to be
            passed the first time the extractor is initialized.
        model (str): The name of the Clarifai model to use. If None, defaults
            to the general image tagger.
        min_value (float): A value between 0.0 and 1.0 indicating the minimum
            confidence required to return a prediction. Defaults to 0.0.
        max_concepts (int): A value between 0 and 200 indicating the maximum
            number of label predictions returned.
        select_concepts (list): List of concepts (strings) to query from the
            API. For example, ['food', 'animal'].
        rate_limit (int): The minimum number of seconds required between
            transform calls on this Transformer.
        batch_size (int): Number of stims to send per batched API request.
    '''

    _batch_size = 32

    def __init__(self, access_token=None, user_id=None, app_id=None, model='general-image-recognition', min_value=None,
                 max_concepts=None, select_concepts=None, rate_limit=None,
                 batch_size=None):
        super().__init__(access_token=access_token,
                            user_id=user_id,
                            app_id=app_id,
                            model=model,
                            min_value=min_value,
                            max_concepts=max_concepts,
                            select_concepts=select_concepts,
                            rate_limit=rate_limit,
                            batch_size=batch_size)

    def _extract(self, stims):
        verify_dependencies(['clarifai_api'])

        # ExitStack lets us use filename context managers simultaneously
        with ExitStack() as stack:
            imgs = []
            for s in stims:
                if s.url:
                    image=clarifai_api.resources_pb2.Image(url=s.url)
                    
                else:
                    f_name = stack.enter_context(s.get_filename())
                    with open(f_name, "rb") as f:
                        file_bytes = f.read()
                    image = clarifai_api.resources_pb2.Image(
                                base64=file_bytes
                            )
                image = clarifai_api.resources_pb2.Input(
                    data=clarifai_api.resources_pb2.Data(
                        image=image
                    )
                )
                imgs.append(image)
            outputs = self._query_api(imgs)

        extractions = []
        for i, resp in enumerate(outputs):
            extractions.append(ExtractorResult(resp, stims[i], self))
        return extractions

    def _to_df(self, result):
        if self.model_name == 'face-detection':
            # is a list already, no need to wrap it in one
            return pd.DataFrame(self._parse_annotations(result._data))
        return pd.DataFrame([self._parse_annotations(result._data)])


class ClarifaiAPIVideoExtractor(ClarifaiAPIExtractor, VideoExtractor):
    ''' Uses the Clarifai API to extract tags from videos.

    Args:
        api_key (str): A valid API_KEY for the Clarifai API. Only needs to be
            passed the first time the extractor is initialized.
        model (str): The name of the Clarifai model to use. If None, defaults
            to the general image tagger.
        min_value (float): A value between 0.0 and 1.0 indicating the minimum
            confidence required to return a prediction. Defaults to 0.0.
        max_concepts (int): A value between 0 and 200 indicating the maximum
            number of label predictions returned.
        select_concepts (list): List of concepts (strings) to query from the
            API. For example, ['food', 'animal'].
        rate_limit (int): The minimum number of seconds required between
            transform calls on this Transformer.
        batch_size (int): Number of stims to send per batched API request.
    '''

    def _extract(self, stim):
        verify_dependencies(['clarifai_api'])
        with stim.get_filename() as filename:
            with open(filename, "rb") as f:
                file_bytes = f.read()
            vids = [clarifai_api.resources_pb2.Input(
                    data=clarifai_api.resources_pb2.Data(
                        video=clarifai_api.resources_pb2.Video(base64=file_bytes)
                    )
                )]
            outputs = self._query_api(vids)
        return ExtractorResult(outputs, stim, self)

    def _to_df(self, result):
        onsets = []
        durations = []
        data = []
        frames = result._data[0].data.frames
        for i, frame_res in enumerate(frames):
            tmp_res = self._parse_annotations(frame_res)
            # if we detect multiple faces, the parsed annotation can be multi-line
            if type(tmp_res) == list:
                for d in tmp_res:
                    data.append(d)
                    onset = frame_res.frame_info.time / 1000.0

                    if (i + 1) == len(frames):
                        end = result.stim.duration
                    else:
                        end = frames[i + 1].frame_info.time / 1000.0
                    onsets.append(onset)
                    durations.append(max([end - onset, 0]))

                    result._onsets = onsets
                    result._durations = durations
                    df = pd.DataFrame(data)
                    result.features = list(df.columns)
            else:
                data.append(tmp_res)
                onset = frame_res.frame_info.time / 1000.0

                if (i + 1) == len(frames):
                    end = result.stim.duration
                else:
                    end = frames[i+1].frame_info.time / 1000.0
                onsets.append(onset)
            # NOTE: As of Clarifai API v2 and client library 2.6.1, the API
            # returns more frames than it should—at least for some videos.
            # E.g., given a 5.5 second clip, it may return 7 frames, with the
            # last beginning at 6000 ms. Since this appears to be a problem on
            # the Clarifai end, and it's not actually clear how they're getting
            # this imaginary frame (I'm guessing it's the very last frame?),
            # we're not going to do anything about it here, except to make sure
            # that durations aren't negative.
                durations.append(max([end - onset, 0]))

            result._onsets = onsets
            result._durations = durations
            df = pd.DataFrame(data)
            result.features = list(df.columns)

        return df
