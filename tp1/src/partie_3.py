# ----------------------------------------------------------------
# Authors: Mathieu Kabore, Florence Gaborit and Reph D. Mombrun
# Date: 11/09/2018
# Last update: 12/09/2018
# INF8215 TP1
# ----------------------------------------------------------------

import time
import copy
import numpy as np
from random import shuffle, randint

from stack import Stack

class Solution:
    def __init__(self, places, graph):
        """
        places: a list containing the indices of attractions to visit
        p1 = places[0]
        pm = places[-1]
        """
        self.g = 0  # current cost
        self.graph = graph
        self.visited = [places[0]]  # list of already visited attractions
        # list of attractions not yet visited
        self.not_visited = copy.deepcopy(places[1:])

    def add(self, idx):
        """
        Adds the point in position idx of not_visited list to the solution
        """
        # TODO : to implement


def initial_sol(graph, places_to_visit):
    """
    Return a completed initial solution
    """
    # TODO : to verify
    return dfs(graph, places_to_visit)


def dfs(graph, places_to_visit):
    """
    Performs a Depth-First Search
    """
    # TODO : to implement instead of returning the following solution instance
    return Solution(places_to_visit, graph)


def shaking(sol, k):
    """
    Returns a solution on the k-th neighrboohood of sol
    """
    # TODO : to implement instead of returning the same solution passed as parameter
    return sol


def local_search_2opt(sol):
    """
    Apply 2-opt local search over sol
    """
    # TODO : to implement instead of returning the same solution passed as parameter
    return sol


def vns(sol, k_max, t_max):
    """
    Performs the VNS algorithm
    """
    # TODO : to implement instead of returning the same solution passed as parameter
    return sol