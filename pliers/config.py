
cache_converters = False
cache_filters = False
cache_extractors = False
log_transformations = True
default_converters = {
    'AudioStim->TextStim': ('IBMSpeechAPIConverter', 'WitTranscriptionConverter')
}