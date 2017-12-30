cache_transformers = True
log_transformations = True
drop_bad_extractor_results = True
progress_bar = False
parallelize = False
use_generators = False
n_jobs = None
default_converters = {
    'AudioStim->TextStim': ('IBMSpeechAPIConverter', 'WitTranscriptionConverter'),
    'ImageStim->TextStim': ('GoogleVisionAPITextConverter', 'TesseractConverter')
}
