""" Tests for motif utilitiy functions."""

from pytest import mark

from ndspflow.motif.utils import motif_to_cycle


@mark.parametrize("ttype", ['euclidean', 'similarity', 'affine', 'projective', 'polynomial'])
def test_motif_to_cycle(motif_outs, ttype):

    motif_ref = motif_outs['motif_ref']
    motif_target = motif_outs['motif_target']

    motif_trans, tform = motif_to_cycle(motif_target, motif_ref, ttype=ttype)

    assert len(motif_trans) == len(motif_ref) == len(motif_target)
    assert tform is not None
