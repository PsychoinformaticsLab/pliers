'''
Extractors that interact with the AWS Rekognition API.
'''

from pliers.stimuli.image import ImageStim
from pliers.extractors.base import Extractor, ExtractorResult
import pandas as pd
import boto3
from pliers.utils import attempt_to_import, verify_dependencies

aws_rekognition_client = attempt_to_import('boto3')


class AwsRekognitionExtractor(ImageStim, Extractor):

    def __init__(self, profile_name=None, region_name=None, extractor_type=None):
        verify_dependencies(['boto3'])
        if profile_name is not None and region_name is not None:
            self.session = boto3.Session(profile_name=profile_name)
            self.rekognition = boto3.Session.client(
                'rekognition', region_name=region_name)

        elif profile_name is not None:
            self.rekognition = boto3.client('rekognition')
            self.session = boto3.Session(profile_name=profile_name)
        else:
            self.rekognition = boto3.client('rekognition')

        self.extractor_type = extractor_type

        super(AwsRekognitionExtractor, self).__init__()

    def _get_value(self, stim):

        img2byte = self.img_to_byte(stim)

        response = getattr(self.rekognition, self.extractor_type)(
            Image={
                'Bytes': img2byte
            }

            #            , Attributes=['ALL']
        )

        return response

    def _extract(self, stim):

        if self.extractor_type is None:
            self.extractor_type = 'detect_face'

        values = self._get_values(stim)

        return ExtractorResult(values, stim, self,
                               features=values,
                               )

    @property
    def img_to_byte(img):
        #     Convert image to bytes
        with img.get_filename() as filename:
            with open(filename, "rb") as image:
                f = image.read()
                b = bytearray(f)
                return(b)

    def _to_df(self, data_dictionary):

        result = self.convert_to_dataframe(data_dictionary)
        obj_df = pd.DataFrame.from_dict(result, orient='index')
        return obj_df

    def convert_to_dataframe(self, d, result=None, index=None, Key=None):
        if result is None:
            result = {}
        if isinstance(d, (list, tuple)):
            for indexB, element in enumerate(d):
                if Key is not None:
                    newkey = Key
                self.to_df(element, result, index=indexB, Key=newkey)
        elif isinstance(d, dict):
            for key in d:
                value = d[key]
                if Key is not None and index is not None:
                    newkey = "_".join(
                        [Key, (str(key).replace(" ", "") + str(index))])
                elif Key is not None:
                    newkey = "_".join([Key, (str(key).replace(" ", ""))])
                else:
                    newkey = str(key).replace(" ", "")
                self.to_df(value, result, index=None, Key=newkey)
        else:
            result[Key] = d
        return result


class DetectFaceAWSExtractor(AwsRekognitionExtractor):

    extractor_type = 'detect_faces'


class CompareFaceAWSExtractor(AwsRekognitionExtractor):

    extractor_type = 'compare_faces'

    def _extract(self, stim, target_stim, threshold=None, **kwargs):

        if threshold is None:
            self.threshold = 70
        else:
            self.threshold = threshold

        values = self._get_values(stim, target_stim)

        return ExtractorResult(values, stim, self,
                               features=values,
                               )
#        super(CompareFaceAWSExtractor, self)._extract( **kwargs)

    def _get_value(self, stim, target_stim):

        source_img2byte = self.img_to_byte(stim)
        target_img2byte = self.img_to_byte(target_stim)

        response = getattr(self.rekognition, self.extractor_type)(
            SourceImage={
                'Bytes': source_img2byte
            },
            TargetImage={
                'Bytes': target_img2byte
            },

            SimilarityThreshold=self.threshold,

        )

        return response


class DetectLabelsAWSExtractor(AwsRekognitionExtractor):

    extractor_type = 'detect_labels'

    def _extract(self, stim, tmax_labels=None, min_confidence=None, **kwargs):

        if tmax_labels is None:
            self.tmax_labels = 10
        else:
            self.tmax_labels = tmax_labels

        if min_confidence is None:
            self.min_confidence = 70
        else:
            self.min_confidence = min_confidence

        values = self._get_values(stim)

        return ExtractorResult(values, stim, self,
                               features=values,
                               )

    def _get_value(self, stim):

        source_img2byte = self.img_to_byte(stim)

        response = getattr(self.rekognition, self.extractor_type)(
            SourceImage={
                'Bytes': source_img2byte
            },
            MaxLabels=self.max_labels,
            MinConfidence=self.min_confidence
        )

        return response


class DetectTextAWSExtractor(AwsRekognitionExtractor):

    extractor_type = 'detect_text'


class DetectModerationLabelsAWSExtractor(AwsRekognitionExtractor):

    extractor_type = 'detect_moderation_labels'

    def _extract(self, stim, min_confidence=None, **kwargs):

        if min_confidence is None:
            self.min_confidence = 50
        else:
            self.min_confidence = min_confidence

        values = self._get_values(stim)

        return ExtractorResult(values, stim, self,
                               features=values,
                               )

    def _get_value(self, stim):

        source_img2byte = self.img_to_byte(stim)

        response = getattr(self.rekognition, self.extractor_type)(
            SourceImage={
                'Bytes': source_img2byte
            },
            MinConfidence=self.min_confidence
        )

        return response


class RecognizeCelebritiesAWSExtractor(AwsRekognitionExtractor):

    extractor_type = 'recognize_celebrities'


class CreateCollectionAWSExtractor(AwsRekognitionExtractor):

    extractor_type = 'create_collection'

    def _extract(self, collection_id, **kwargs):
        self.collection_id = collection_id

        values = self._get_values()

        return ExtractorResult(values, collection_id, self,
                               features=values,
                               )

    def _get_value(self):
        response = getattr(self.rekognition, self.extractor_type)(
            CollectionId=self.collection_id
        )

        return response


class IndexFacesAWSExtractor(AwsRekognitionExtractor):

    extractor_type = 'index_faces'

    def _extract(self, stim, collection_id, **kwargs):

        self.collection_id = collection_id

        values = self._get_values(stim)

        return ExtractorResult(values, stim, self,
                               features=values,
                               )

    def _get_value(self, stim):

        source_img2byte = self.img_to_byte(stim)

        response = getattr(self.rekognition, self.extractor_type)(
            SourceImage={
                'Bytes': source_img2byte
            },

            CollectionId=self.collection_id
        )

        return response


class ListFacesAWSExtractor(AwsRekognitionExtractor):

    extractor_type = 'list_faces'

    def _extract(self, collection_id, **kwargs):

        self.collection_id = collection_id

        values = self._get_values()

        return ExtractorResult(values, collection_id, self,
                               features=values,
                               )

    def _get_value(self):
        response = getattr(self.rekognition, self.extractor_type)(
            CollectionId=self.collection_id
        )

        return response


class SearchFacesAWSExtractor(AwsRekognitionExtractor):

    extractor_type = 'list_faces'

    def _extract(self, collection_id, face_id, max_faces=None, **kwargs):

        self.collection_id = collection_id
        self.face_id = face_id

        self.max_faces = max_faces

        values = self._get_values()

        return ExtractorResult(values, face_id, self,
                               features=values,
                               )

    def _get_value(self):

        if self.max_faces is not None:

            response = getattr(self.rekognition, self.extractor_type)(
                CollectionId=self.collection_id,
                FaceId=self.face_id,
                MaxFaces=self.max_faces
            )
        else:

            response = getattr(self.rekognition, self.extractor_type)(
                CollectionId=self.collection_id,
                FaceId=self.face_id
            )

        return response


class SearchFacesByImageAWSExtractor(AwsRekognitionExtractor):

    extractor_type = 'search_faces_by_image'

    def _extract(self, stim, collection_id, face_match_threshold=None, max_faces=None, **kwargs):

        self.collection_id = collection_id
        self.face_match_threshold = face_match_threshold

        self.max_faces = max_faces

        values = self._get_values(stim)

        return ExtractorResult(values, stim, self,
                               features=values,
                               )

    def _get_value(self, stim):

        source_img2byte = self.img_to_byte(stim)

        if self.max_faces is not None:

            response = getattr(self.rekognition, self.extractor_type)(
                CollectionId=self.collection_id,
                SourceImage={
                    'Bytes': source_img2byte
                },
                MaxFaces=self.max_faces
            )
        elif self.face_match_threshold is not None:

            response = getattr(self.rekognition, self.extractor_type)(
                CollectionId=self.collection_id,
                SourceImage={
                    'Bytes': source_img2byte
                },
                FaceMatchThreshold=self.face_match_threshold
            )

        elif self.face_match_threshold is not None and self.max_faces is not None:

            response = getattr(self.rekognition, self.extractor_type)(
                CollectionId=self.collection_id,
                SourceImage={
                    'Bytes': source_img2byte
                },
                FaceMatchThreshold=self.face_match_threshold,
                MaxFaces=self.max_faces
            )
        else:

            response = getattr(self.rekognition, self.extractor_type)(
                CollectionId=self.collection_id,
                SourceImage={
                    'Bytes': source_img2byte
                }
            )

        return response
