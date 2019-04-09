''' Rev.ai API-based Converter classes '''

import logging
import os
import time

from pliers.stimuli.text import TextStim, ComplexTextStim
from pliers.utils import attempt_to_import, verify_dependencies
from pliers.converters.audio import AudioToTextConverter
from pliers.transformers.api import APITransformer

rev_ai = attempt_to_import('rev_ai')
rev_ai_client = attempt_to_import('rev_ai.apiclient',
                                  'rev_ai_client',
                                  ['RevAiAPIClient'])


class RevAISpeechAPIConverter(APITransformer, AudioToTextConverter):

    ''' Uses the Rev AI speech-to-text API to transcribe an audio file.

    Args:
        access_token (str): API credential access token. Must be passed
            explicitly or stored in the environment variable specified
            in the _env_keys field.
        timeout (int): Number of seconds to wait for audio transcription
            to finish. Defaults to 90 seconds.
        request_rate (int): Number of seconds to wait between polling the
            API for completion.
    '''

    _env_keys = ('REVAI_ACCESS_TOKEN',)
    _log_attributes = ('access_token', 'timeout', 'request_rate')
    VERSION = '1.0'

    def __init__(self, access_token=None, timeout=1000, request_rate=5):
        verify_dependencies(['rev_ai_client'])
        if access_token is None:
            try:
                access_token = os.environ['REVAI_ACCESS_TOKEN']
            except KeyError:
                raise ValueError("A valid API key must be passed when a "
                                 "RevAISpeechAPIConverter is initialized.")
        self.access_token = access_token
        self.timeout = timeout
        self.request_rate = request_rate
        self.client = rev_ai_client.RevAiAPIClient(access_token)
        super(RevAISpeechAPIConverter, self).__init__()

    @property
    def api_keys(self):
        return [self.access_token]

    def check_valid_keys(self):
        try:
            account = self.client.get_account()
            if account.balance_seconds > 0:
                return True
            else:
                logging.warn("Insufficient balance for Rev.ai speech "
                             "converter: {}".format(account.balance_seconds))
                return False
        except Exception as e:
            logging.warn(str(e))
            return False

    def _convert(self, audio):
        verify_dependencies(['rev_ai'])
        msg = "Beginning audio transcription with a timeout of %fs. Even for "\
              "small audios, full transcription may take awhile." % self.timeout
        logging.warning(msg)

        if audio.url:
            job = self.client.submit_job_url(audio.url)
        else:
            with audio.get_filename() as filename:
                job = self.client.submit_job_local_file(filename)

        operation_start = time.time()
        response = self.client.get_job_details(job.id)
        while (response.status == rev_ai.JobStatus.IN_PROGRESS) and \
              (time.time() - operation_start) < self.timeout:
            response = self.client.get_job_details(job.id)
            time.sleep(self.request_rate)

        if (time.time() - operation_start) >= self.timeout:
            msg = "Conversion reached the timeout limit of %fs." % self.timeout
            logging.warning(msg)

        if response.status == rev_ai.JobStatus.FAILED:
            raise Exception('API failed: %s' % response.failure_detail)

        result = self.client.get_transcript_object(job.id)

        elements = []
        order = 0
        for m in result.monologues:
            for e in m.elements:
                if e.type_ == 'text':
                    start = e.timestamp
                    end = e.end_timestamp
                    elements.append(TextStim(text=e.value,
                                             onset=start,
                                             duration=end-start,
                                             order=order))
                    order += 1

        return ComplexTextStim(elements=elements, onset=audio.onset)
