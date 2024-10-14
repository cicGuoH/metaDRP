# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:09:36 2023

@author: GUI
"""

import numpy as np
from scipy.sparse import csgraph
import copy

def floyd_warshall(adj_matrix):
    
    dist_matrix, predecessors = csgraph.floyd_warshall(adj_matrix, directed=False, 
                                       return_predecessors=True, unweighted=True,
                                       overwrite=False)
    
    dist_matrix[dist_matrix==np.Inf] = 62 #设置不可到达的两个原子之间距离为64
    predecessors[dist_matrix>=62] = 62
    predecessors[dist_matrix==0] = -1 #对角线为-1
    return dist_matrix, predecessors


def get_all_edges(predecessors, source, target):
    if source == target:
        return []
    prev = predecessors[source]
    curr = prev[target]
    path = [target, curr]
    while curr != source:
        curr = prev[curr]
        path.append(curr)
    return list(reversed(path))

def gen_edge_input(max_dist, predecessors, edge_attr):
    n = predecessors.shape[0]
    edge_attr_all = -1 * np.ones([n, n, max_dist, edge_attr.shape[-1]], dtype=np.int64)
    path_copy = copy.deepcopy(predecessors)
    edge_attr_copy = copy.deepcopy(edge_attr)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path_copy[i][j] == 62:
                continue
            
            path = get_all_edges(path_copy, i, j)
            num_path = len(path) -1 
            for k in range(num_path):
                edge_attr_all[i, j, k, :] = edge_attr_copy[path[k], path[k+1], :]

    return edge_attr_all
