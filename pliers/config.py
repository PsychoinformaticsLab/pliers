
cache_converters = False
cache_filters = False
cache_extractors = False
log_transformations = True
drop_bad_extractor_results = True
default_converters = {
    'AudioStim->TextStim': ('IBMSpeechAPIConverter', 'WitTranscriptionConverter'),
    'ImageStim->TextStim': ('GoogleVisionAPITextConverter', 'TesseractConverter')
}
