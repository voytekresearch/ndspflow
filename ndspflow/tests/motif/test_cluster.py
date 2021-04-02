"""Tests for clustering cycles."""

import numpy as np

from neurodsp.sim import sim_cycle

from ndspflow.motif import cluster_cycles


def test_cluster_cycles():

    cyc = sim_cycle(0.1, 1000, 'sine')
    cyc_inv = cyc * -1
    cycles = np.array([*[cyc]*5, *[cyc_inv]*5])

    # Two clusters
    labels = cluster_cycles(cycles, clust_score=0.5, max_clusters=11)
    assert all([lab == labels[0] for lab in labels[1:5]])
    assert all([lab == labels[5] for lab in labels[6:]])

    # No clusters
    labels = cluster_cycles(cycles, clust_score=1.1)
    assert np.isnan(labels)

    # Invalid param passed to KMeans
    labels = cluster_cycles(cycles, clust_score=0.5, min_clusters=-1, max_clusters=0)
    assert np.isnan(labels)

    # Nothing to cluster
    labels = cluster_cycles(cycles, max_clusters=1)
    assert np.isnan(labels)
