'''
Extractors that interact with the AWS Rekognition API.
'''

from pliers.stimuli.image import ImageStim
from pliers.extractors.base import Extractor, ExtractorResult
import numpy as np
import pandas as pd
import boto3
from pliers.utils import attempt_to_import, verify_dependencies

aws_rekognition_client= attempt_to_import('boto3')


class AwsRekognitionExtractor():
    
    def __init__(self,profile_name='default', region_name='us-east-1' ):
            verify_dependencies(['boto3'])
            self.session = boto3.Session(profile_name='user2')
            self.rekognition = boto3.Session.client('rekognition', region_name='us-east-1')
            super(AwsRekognitionExtractor, self).__init__()
    
    
    def _extract(self, stim):
        img2byte=self.img_to_byte(stim)
        response=self.rekognition.detect_faces(
            Image={
                'Bytes': img2byte
                },
                    
            Attributes=['ALL']
        )
        return response
    
    
    @property
    def img_to_byte(img):
    #     Convert image to bytes
        with img.get_filename() as filename:
            with open(filename, "rb") as image:
                f = image.read()
                b = bytearray(f)
                return(b)
                
   
    def to_df(self, d,result=None,index=None,Key=None):
        if result is None:
            result = {}
        if isinstance(d, (list, tuple)):
            for indexB, element in enumerate(d):
                if Key is not None:
                    newkey = Key
                self.to_df(element,result,index=indexB,Key=newkey)            
        elif isinstance(d, dict):        
            for key in d:
                value = d[key]         
                if Key is not None and index is not None:
                    newkey = "_".join([Key,(str(key).replace(" ", "") + str(index))])
                elif Key is not None:
                    newkey = "_".join([Key,(str(key).replace(" ", ""))])
                else:
                    newkey= str(key).replace(" ", "")
                self.to_df(value,result,index=None,Key=newkey)        
        else:
            result[Key]=d 
            
        obj_df=pd.DataFrame.from_dict(result, orient='index')
        return obj_df




#for k,v in new_obj.items():
#    print(k,' = ', v)
#    print('\n')

#img_path='./image.jpeg'
#
#img=ImageStim(filename=img_path)
#print(img)
#new_obj =  to_df(response)
#print(new_obj)