"""Cluster cycles into multiple motifs at the same frequency."""

import warnings

import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_cycles(cycles, min_clust_score=0.5, min_clusters=2, max_clusters=10, random_state=None):
    """K-means clustering of cycles.

    Parameters
    ----------
    cycles : 2D array
        Cycles within a frequency range.
    min_clust_score : float, optional, default: 0.5
        The minimum silhouette score to accept k clusters..
    min_clusters : int, optional, default: 2
        The minimum number of clusters to evaluate.
    max_clusters : int, optional, default: 10
        The maximum number of clusters to evaluate.
    random_state : int, optional, default: None
        Determines random number generation for centroid initialization.
        Use an int to make the randomness deterministic for reproducible results.

    Returns
    -------
    labels : 1d array
        The predicted cluster each cycle belongs to.
    """

    # Nothing to cluster
    if len(cycles) == 1 or max_clusters == 1:
        return np.nan

    max_clusters = len(cycles) if len(cycles) < max_clusters else max_clusters

    labels = []
    scores = []

    for n_clusters in range(min_clusters, max_clusters+1):

        if n_clusters == 1:
            continue

        if n_clusters > len(cycles) - 1:
            break

        with warnings.catch_warnings():

            # Skip to next cluster definition if k-means fails
            try:
                clusters = KMeans(n_clusters=n_clusters, algorithm="full",
                                  random_state=random_state)
                clusters = clusters.fit_predict(cycles)
            except:
                continue

        labels.append(clusters)
        scores.append(silhouette_score(cycles, clusters))

    # No superthreshold clusters found
    if len(scores) < 1 or max(scores) < min_clust_score:
        return np.nan

    # Split motifs based on highest silhouette score
    labels = labels[np.argmax(scores)]

    return labels
