"""Test input/output utility functions"""

import os
from tempfile import TemporaryDirectory

import pytest

from ndspflow.tests.settings import TEST_DATA_PATH
from ndspflow.io.paths import check_dirs


@pytest.mark.parametrize(
    "input_dir", [
        TEST_DATA_PATH,
        pytest.param(False, marks=pytest.mark.xfail(raises=ValueError))
    ],
)
@pytest.mark.parametrize("output_dir_exists", [True, False])
def test_check_dirs(input_dir, output_dir_exists):

    if output_dir_exists:
        temp_dir = TemporaryDirectory()
        output_dir = temp_dir.name
    else:
        output_dir = os.path.join(TEST_DATA_PATH, 'test_dir')

    check_dirs(input_dir, output_dir)

    if output_dir_exists:
        temp_dir.cleanup()
    else:
        os.rmdir(output_dir)
