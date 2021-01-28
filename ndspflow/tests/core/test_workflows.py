"""Test nipype workflows"""

from tempfile import TemporaryDirectory
from pytest import mark, param

import nipype.pipeline.engine as pe

from ndspflow.core.workflows import create_workflow, wf_fooof, wf_bycycle
from ndspflow.tests.settings import TEST_DATA_PATH
from ndspflow.core.interfaces import FOOOFNode, BycycleNode


def test_create_workflow():

    input_dir = TEST_DATA_PATH
    temp_dir = TemporaryDirectory()
    output_dir = temp_dir.name

    fooof_params={'freqs': 'freqs.npy', 'power_spectrum': 'powers.npy'}
    bycycle_params = {'sig': 'sig.npy', 'fs': 500, 'f_range_bycycle': (1, 40)}

    wf = create_workflow(input_dir, output_dir, run_nodes=['fooof', 'bycycle'],
                         fooof_params=fooof_params, bycycle_params=bycycle_params)

    assert isinstance(wf, pe.workflows.Workflow)
    assert wf._name == 'wf_ndspflow'

    temp_dir.cleanup()


@mark.parametrize("fooof_params", [{'freqs': 'freqs.npy', 'power_spectrum': 'powers.npy'},
                                   param(None, marks=mark.xfail), param({}, marks=mark.xfail)])
def test_wf_fooof(fooof_params):

    wf = wf_fooof(fooof_params)

    assert wf._name == 'fooof_node'
    assert isinstance(wf._interface, FOOOFNode)


@mark.parametrize("bycycle_params", [{'sig': 'sig.npy', 'fs': 500, 'f_range_bycycle': (1, 40)},
                                     param(None, marks=mark.xfail), param({}, marks=mark.xfail)])
def test_wf_bycycle(bycycle_params):

    wf = wf_bycycle(bycycle_params)

    assert wf._name == 'bycycle_node'
    assert isinstance(wf._interface, BycycleNode)
