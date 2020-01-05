# -*- coding: utf-8 -*-
"""
Grid-DBSCAN in GC-net
based on Scipy & scikit
https://www.scipy.org/
https://scikit-learn.org
add it into scikit-learn/sklearn/cluster/
"""

import numpy as np
from scipy import sparse
 
from ..base import BaseEstimator, ClusterMixin
from ..utils import check_array
from ..utils.validation import _check_sample_weight
from ..neighbors import NearestNeighbors

from .cython import gdn_inner

    """
    Main Parameters
    ----------
    X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
            array of shape (n_samples, n_samples)
        A feature array, or array of distances between samples if
        ``metric='precomputed'``.
        X in point clouds is initialized to X[x,y,z,id,azimuth]
    eps : float, optional
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.
    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    Neb : float, array
        Size of the neighborhood of index.
        algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        The algorithm to be used by the NearestNeighbors module
        to compute pointwise distances and find nearest neighbors.
        See NearestNeighbors module documentation for details.
        Initialized to brute, which may have worst case O(n^2) 
        memory complexity.
    p : float, optional
        The power of the Minkowski metric to be used to calculate distance
        between points. 
        Initialized to 2, which means the Euclidean distance. 
    
    Other parameters are initialized to the same value as the conventional DBSCAN.
    """

class DBSCAN(ClusterMixin, BaseEstimator):
    def __init__(self, eps, min_samples, Neb, metric='euclidean',
                 metric_params=None, algorithm='auto', leaf_size=30, p=2,
                 n_jobs=None):
        self.eps = eps
        self.min_samples = min_samples
        self.neb = Neb
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    def fit(self, X, y=None, sample_weight=None):

        X = check_array(X, accept_sparse='csr')
        
        if self.metric == 'precomputed' and sparse.issparse(X):
            # set the diagonal to explicit values, as a point is its own
            # neighbor
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)
                X.setdiag(X.diagonal())  # XXX: modifies X's internals in-place
                
        neighbors_model = NearestNeighbors(
            radius=self.eps, algorithm=self.algorithm,
            leaf_size=self.leaf_size, metric=self.metric,
            metric_params=self.metric_params, p=self.p, n_jobs=self.n_jobs)            
        c=0            
        for i in X:
            index_neighbor = asarray(
                                     (X[3]>=(i[3]-neb[0]))&(X[3]<=(i[3]+neb[0]))
                                     &(X[4]>=(i[4]-neb[1]))&(X[4]<=(i[4]+neb[1]))
                                     )
            if c==0:
                index_neighborhoods = index_neighbor
                c+=1
            else:
                np.append(index_neighborhoods, index_neighbor, axis=0)
        
        for x in index_neighborhoods:
            neighbors_model.fit(x[:,(0,1,2)])            
            neighborhoods = neighbors_model.radius_neighbors(x[:,(0,1,2)],
                                                         return_distance=False)

        n_neighbors = np.array([np.sum(sample_weight[neighbors])
                                for neighbors in neighborhoods])

        # Initially, all samples are noise.
        labels = np.full(X.shape[0], -1, dtype=np.intp)

        # A list of all core samples found.
        core_samples = np.asarray(n_neighbors >= self.min_samples,
                                  dtype=np.uint8)
        gdn_main(core_samples, neighborhoods, labels)

        self.core_sample_indices_ = np.where(core_samples)[0]
        self.labels_ = labels

        if len(self.core_sample_indices_):
            # fix for scipy sparse indexing issue
            self.components_ = X[self.core_sample_indices_].copy()
        else: 
            # no core samples
            self.components_ = np.empty((0, X.shape[1]))
        return self

    def fit_predict(self, X, y=None, sample_weight=None):

        self.fit(X, sample_weight=sample_weight)
        return self.labels_
