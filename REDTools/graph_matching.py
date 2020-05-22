""" Graph Matching

A Number of functions for matching graphs, calculating density
and generating permuted null distributions for permutation testing

Authors: Alex Anwyl-Irvine & Edwin Dalmaijer 2020

airvine1991@gmail.com

"""

__all__ = ['calc_density', 'permute_connections', 'match_density', 'match_graphs_participant', 'match_graphs', 'permute_and_match', 'generate_null_dist']
__version__ = '0.1'
__author__ = 'Alex Anwyl-Irvine & Edwin Dalmaijer'

import numpy as np
import scipy.stats as ss
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
import joblib


def calc_density(matrix):
    """returns the density of a given matrix

    This returns the density of a given matrix, excluding the diagonals

    :param matrix: the matrix itself
    :return: the density value, 0-1
    """
    rem_self =  matrix- np.diag(np.diag(matrix))
    return np.count_nonzero(rem_self)/np.prod(rem_self.shape)

def permute_connections(m):

    """Permutes the connection weights in a sparse 2D matrix. Keeps the
    location of connections in place, but permutes their strengths.

    Arguments

    m       -   numpy.ndarray with a numerical dtype and shape (N,N) where N
                is the number of nodes. Lack of connection should be denoted
                by 0, and connections should be positive or negative values.

    Returns

    perm_m  -   numpy.ndarray of shape (N,N) with permuted connection weights.
    """

    # Verify the input.
    if len(m.shape) != 2:
        raise Exception("Connection matrix `m` should be 2D")
    if m.shape[0] != m.shape[1]:
        raise Exception("Connection matrix `m` should be symmetric")

    # Copy the original matrix to prevent modifying it in-place.
    perm_m = np.copy(m)

    # Get the indices to the lower triangle.
    i_low = np.tril_indices(perm_m.shape[0], -1)

    # Create a flattened copy of the connections.
    flat_m = perm_m[i_low]

    # Shuffle all non-zero connections.
    nonzero = flat_m[flat_m != 0]
    np.random.shuffle(nonzero)
    flat_m[flat_m != 0] = nonzero

    # Place the shuffled connections over the original connections in the
    # copied matrix.
    perm_m[i_low] = flat_m

    # Copy lower triangle to upper triangle to make the matrix symmertical.
    perm_m.T[i_low] = perm_m[i_low]

    return perm_m

def match_density(sample_matrix, target_matrix, start_t, step, iterations):
    """ Iteratively removes connections of matrix to match density

    :param sample_matrix: The matrix we are adjusting to match
    :param target_matrix: The matrix we want to match density to
    :param start_t: The starting threshold for removing connections
    :param step: The step to iteratively adjust threshold
    :param iterations: Number of iterations before giving up
    :return: the density adjusted sample matrix
    """
    target_densities = [calc_density(f) for f in target_matrix]


    # target density to match input
    target_density = np.mean(target_densities)
    print(f'target density of: {target_density}')
    print(f'now iterating {iterations} times')
    for i in range(iterations): # iterations
        tmp_c = sample_matrix.copy() # make copy
        for ii in range(sample_matrix.shape[0]): # participants
            tmp_c[ii][tmp_c[ii,:,:] < start_t] = 0
        # calc density
        mean_density = np.mean([calc_density(f) for f in tmp_c])

        if mean_density > target_density:
            start_t += step
        else:
            start_t -= step

    print(f'reached density of: {mean_density}')
    print(f'this is {target_density - mean_density} away from target')
    return tmp_c

def match_graphs_participant(graph_1, graph_2, p):
    """ Matches graphs for one participant only

    The process is as follows:
        - Z-score both graphs
        - Remove infinite and NaN values
        - Calculate euclidean distance between all nodes in two graphs, based on connections
        - Use linear sum assignment algorithm to match based on euclidean distance (as a cost function)

    :param graph_1: The first graph for matching, all participants
    :param graph_2: The second graph for matching, all participants
    :param p: The participant

    :return: binary_nodes, binary_match_mat, cost_euc:

    binary nodes: each nodes self matching accuracy - so 1D array
    binary_match_mat: binary matching matrix showing which node each node matched to

    """


    print(p)

    # Here we just use one participant
    this_structural = graph_1[p, :,:]
    this_functional = graph_2[p,:,:]


    # z score both
    this_structural = ss.zscore(this_structural)
    this_functional = ss.zscore(this_functional)

    # deal with NaN and infinte values
    this_structural = np.nan_to_num(this_structural)
    this_functional = np.nan_to_num(this_functional)
    # Step 1, feature vector for each node, containing connectivity for each over node in graph
    # Note: this is essentially the connectome matrix (each row is a feature vector), so no need to do anything here

    # Step 2, cost function for each node in graph 1 to every node in functional graph 2
    # Cost funcion is the eucledian distance between the two feature vectors of these nodes, do this for all nodes
    # structural
    cost_euc = np.empty(this_structural.shape)
    for i in range(cost_euc.shape[0]):
        cost_euc[i] = [euclidean(this_structural[i], f) for f in this_functional]

    # Step 3, Hungarian algorithm for matching nodes of structural graphs with that of functional graphs,
    # Matching matrix where each structural node has it's most similiar equivalent node in functional
    match_inds = linear_sum_assignment(cost_euc) # 2nd index gives least cost cost node for each node

    # Step 4, Create a binary matching accuracy graph. A node being matched with it's equivalent in the other graph
    # is considered accurate (i.e. a 1) anything else is a 0
    binary_nodes = np.array([i == j for i, j in zip(match_inds[0], match_inds[1])])

    # Make this in 2D as well
    binary_match_mat = np.zeros(this_structural.shape, dtype=int) # intialise empty
    for i, row in enumerate(binary_match_mat): # loop through rows/nodes
        row[match_inds[1][i]] = 1 # place a 1 in the corresponding column where it matched


    return binary_nodes, binary_match_mat, cost_euc

def match_graphs(graph_1, graph_2, njobs):
    """ Takes two graphs and performs inexact graph matching


    :param graph_1:
    :param graph_2:
    :param njobs:
    :return:
    """

    output = \
        joblib.Parallel(n_jobs=njobs)(joblib.delayed(match_graphs_participant)(graph_1, graph_2, p) for p in range(graph_1.shape[0]))

    binary_matched = np.array([f[0] for f in output])
    binary_matched_matrix = np.array([f[1] for f in output])
    euc_distances = np.array([f[2] for f in output])
    return binary_matched, binary_matched_matrix, euc_distances

def permute_and_match(graph_1, graph_2, njobs):
    """ performs a single permutation and a match

    :param graph_1: group level graph 1
    :param graph_2: group level graph 2
    :param njobs: number of jobs for parallelisation
    :return:
    """
    # permute one graph
    graph_1_p = np.array([permute_connections(g) for g in graph_1])
    return match_graphs(graph_1_p, graph_2, njobs)[1]

def generate_null_dist(graph_1, graph_2, perms, njobs):
    """ Generates a matrix of results from permuted matrix comparisons
    :param graph_1: graph for matching
    :param graph_2: graph for matching
    :param perms: permutations to perform
    :param njobs: how many parallel jobs to use
    :return: matrix of permuted results per participant and per connection
    """

    bin_matrix = np.zeros([graph_1.shape[1], graph_1.shape[2], perms]) # for the nulls
    for p in range(perms): # each permutation
        perm_matrix = permute_and_match(graph_1,graph_2, njobs) # get permuted matches
        bin_matrix[:,:,p] = perm_matrix.mean(axis=0) # calculate the distribution metric

    return bin_matrix
