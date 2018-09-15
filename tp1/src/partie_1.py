# ----------------------------------------------------------------
# Authors: Mathieu Kabore, Florence Gaborit and Reph D. Mombrun
# Date: 11/09/2018
# Last update: 12/09/2018
# INF8215 TP1
# ----------------------------------------------------------------

import numpy as np
import copy
import time

try:
    import Queue as q
except ImportError:
    import q

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
        s=Solution(places, graph)
        s.visited=self.visited
        s.not_visited=self.not_visited
        s.g = self.g
        s.g +=graph[s.visited[-1]][self.not_visited[idx]]
        s.visited.append(self.not_visited[idx])
        del self.not_visited[idx]
        return s
       
        """
        Adds the point in position idx of not_visited list to the solution
        """


def bfs(graph, places):
    
    """
    Returns the best solution which spans over all attractions indicated in 'places'
    """
    s0=Solution(places, graph)
    q = Queue()
    q.push(s0)
    
    
    return Solution(places, graph)
    
    
    