"""Test nipype workflows"""

from tempfile import TemporaryDirectory
import nipype.pipeline.engine as pe

from ndspflow.core.workflows import create_workflow, wf_fooof
from ndspflow.tests.settings import TEST_DATA_PATH
from ndspflow.core.interfaces import FOOOFNode


def test_create_workflow():

    input_dir = TEST_DATA_PATH
    temp_dir = TemporaryDirectory()
    output_dir = temp_dir.name

    fooof_params={'freqs': 'freqs.npy', 'power_spectrum': 'powers.npy'}
    wf = create_workflow(input_dir, output_dir, run_nodes=['fooof'],
                         fooof_params=fooof_params)

    assert type(wf) is pe.workflows.Workflow
    assert wf._name is 'wf_ndspflow'

    temp_dir.cleanup()


def test_wf_fooof():

    fooof_params={'freqs': 'freqs.npy', 'power_spectrum': 'powers.npy'}
    wf = wf_fooof(fooof_params)

    assert wf._name == 'fooof_node'
    assert type(wf._interface) is FOOOFNode
