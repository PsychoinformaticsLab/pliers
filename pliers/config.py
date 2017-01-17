cache_transformers = True
log_transformations = True
drop_bad_extractor_results = True
progress_bar = True
default_converters = {
    'AudioStim->TextStim': ('IBMSpeechAPIConverter', 'WitTranscriptionConverter'),
    'ImageStim->TextStim': ('GoogleVisionAPITextConverter', 'TesseractConverter')
}
