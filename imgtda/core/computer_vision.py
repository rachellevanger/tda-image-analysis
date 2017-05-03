# TDA Image Analysis Project : core.computer_vision
#
# Copyright (C) 2016-2017 TDA Image Analysis Project
# Author:  Rachel Levanger <rachel.levanger@gmail.com>

"""
Standard numerical techniques for working with image data.
"""

import numpy as np
from scipy import spatial


def match_points(current_points, prior_points, distance_cutoff):
    """
    Takes in an nxd input vector of d-dimensional Euclidean coordinates representing the current dataset
    and an mxd input vector of d-dimensional Euclidean cooridnates representing the prior dataset. 
    Output gives, for each row in _current_points, an mx2 vector that gives
      - the index number from _prior_points in the first column, 
      - and distance matched in second.
    If the matched distance is greater than the cutoff, consider the pair unmatched.
    Unmatched points get -1 for the "matched" index number and the cutoff value for the distance (infinity).
    """

    # Initialize matched indices to -1 and distances to the cutoff value.
    matches = np.ones((current_points.shape[0], 2))*-1
    matches[:,1] = distance_cutoff

    # Initialize index numbers for current points
    current_idx = np.asarray(range(current_points.shape[0]))
    prior_idx = np.asarray(range(prior_points.shape[0]))

    # Generate kd trees
    curr_kd_tree = spatial.KDTree(current_points)
    prior_kd_tree = spatial.KDTree(prior_points)

    # Compute closest keypoint from current->prior and from prior->current
    matches_a = prior_kd_tree.query(current_points)
    matches_b = curr_kd_tree.query(prior_points)

    # Mutual matches are the positive matches within the distance cutoff. All others unmatched.
    potential_matches = matches_b[1][matches_a[1]]
    matched_indices = np.equal(potential_matches, current_idx)

    # Filter out matches that are more than the distance cutoff away.
    in_bounds = (matches_a[0] <= distance_cutoff)
    matched_indices = np.multiply(matched_indices, in_bounds)

    # Add the matching data to the output
    matches[current_idx[matched_indices],0] = prior_idx[matches_a[1]][matched_indices].astype(np.int)
    matches[current_idx[matched_indices],1] = matches_a[0][matched_indices]

    return matches



