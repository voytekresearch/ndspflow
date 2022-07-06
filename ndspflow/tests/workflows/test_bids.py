"""Test BIDS interface."""

import pytest
from shutil import rmtree
from ndspflow.workflows import BIDS


@pytest.mark.parametrize('sub_fs',
    [
        (None, None),
        (['01'], 1000),
        pytest.param((['01'], 100), marks=pytest.mark.xfail)
    ]
)
@pytest.mark.parametrize('allow_ragged', [True, False])
def test_bids(bids_path, sub_fs, allow_ragged):

    subjects, fs = sub_fs

    bids = BIDS(bids_path, subjects=subjects, fs=fs, session='01',
                datatype='ieeg', task='test', extension='.vhdr')

    assert bids.subjects[0] == '01'

    bids.read_bids(subject='01', queue=False, allow_ragged=allow_ragged)

    # Known shape from conftest
    assert len(bids.y_array) == 2
    assert len(bids.y_array[0]) == 2000
    assert bids.fs == 1000

    # No queue, without subject specifier
    bids.read_bids(queue=False, allow_ragged=allow_ragged)

    # Queue
    assert len(bids.nodes) == 0
    bids.read_bids(queue=True, allow_ragged=allow_ragged)
    assert len(bids.nodes) == 1
