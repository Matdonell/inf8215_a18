# ----------------------------------------------------------------
# Authors: Mathieu Kabore, Florence Gaborit and Reph D. Mombrun
# Date: 11/09/2018
# INF8215 TP1
# ----------------------------------------------------------------

import numpy as np
import copy
import time

try:
    import Queue as q
except ImportError:
    import q

from utility import Utility

montreal_graph = Utility.read_graph()


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


def bfs(graph, places):
    """
    Returns the best solution which spans over all attractions indicated in 'places'
    """
    # TODO : to implement
    return Solution(places, graph)


# test 1  --------------  OPT. SOL. = 27
start_time = time.time()
places = [0, 5, 13, 16, 6, 9, 4]
sol = bfs(graph=montreal_graph, places=places)
print(sol.g)
print("--- %s seconds ---" % (time.time() - start_time))

# test 2 -------------- OPT. SOL. = 30
start_time = time.time()
places = [0, 1, 4, 9, 20, 18, 16, 5, 13, 19]
sol = bfs(graph=montreal_graph, places=places)
print(sol.g)
print("--- %s seconds ---" % (time.time() - start_time))

# test 3 -------------- OPT. SOL. = 26
start_time = time.time()
places = [0, 2, 7, 13, 11, 16, 15, 7, 9, 8, 4]
sol = bfs(graph=montreal_graph, places=places)
print(sol.g)
print("--- %s seconds ---" % (time.time() - start_time))
