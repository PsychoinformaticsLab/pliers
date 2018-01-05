import pliers
import tempfile
import os
import json
import pytest
from pliers.config import reset_options


def test_load_from_standard_paths():

    # Verify defaults
    reset_options(update_from_file=False)
    assert pliers.config._settings == pliers.config._default_settings

    # Verify that PLIERS_CONFIG and local dir take precedence
    env_config = {"n_jobs": 200, "log_transformations": False}
    cwd_config = {"log_transformations": True, "parallelize": True}

    handle, f = tempfile.mkstemp(suffix='.json')
    json.dump(env_config, open(f, 'w'))
    os.environ['PLIERS_CONFIG'] = f

    target = 'pliers_config.json'
    if os.path.exists(target):
        pytest.skip("Cannot test pliers config because the default config file"
                    " (pliers_config.json) already exists in the current "
                    "working directory. Skipping test to avoid overwriting.")
    json.dump(cwd_config, open(target, 'w'))

    reset_options(True)
    os.unlink(target)
    opts = pliers.config._settings

    assert opts['n_jobs'] == 200
    assert opts['log_transformations']
    assert opts['parallelize']

    reset_options(False)


def test_set_option():

    reset_options(False)
    opts = pliers.config._settings

    pliers.config.set_options(n_jobs=100, progress_bar=True)
    assert opts['n_jobs'] == 100
    assert opts['progress_bar']

    with pytest.raises(ValueError):
        pliers.config.set_option('bad_key', False)

    reset_options(False)


def test_get_option():
    reset_options(False)
    assert not pliers.config.get_option('parallelize')
