# ----------------------------------------------------------------
# Authors: Mathieu Kabore, Florence Gaborit and Reph D. Mombrun
# Date: 11/09/2018
# INF8215 TP1
# ----------------------------------------------------------------

import time
import copy
import numpy as np
from random import shuffle, randint


def read_graph():
    return np.loadtxt("montreal", dtype='i', delimiter=',')

montreal_graph = read_graph()


# Mini implementation of a stack data structure in python
# https://docs.python.org/3.1/tutorial/datastructures.html
# https://stackoverflow.com/questions/4688859/stack-data-structure-in-python
# http://openbookproject.net/thinkcs/python/english3e/stacks.html
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def is_empty(self):
        return self.items == []


class Solution:
    def __init__(self, places, graph):
        """
        places: a list containing the indices of attractions to visit
        p1 = places[0]
        pm = places[-1]
        """
        self.g = 0 # current cost
        self.graph = graph
        self.visited = [places[0]] # list of already visited attractions
        self.not_visited = copy.deepcopy(places[1:]) # list of attractions not yet visited

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

# test 1  --------------  OPT. SOL. = 27
places = [0, 5, 13, 16, 6, 9, 4]
sol = initial_sol(montreal_graph, places)
start_time = time.time()
vns_sol = vns(sol=sol, k_max=10, t_max=1)
print(vns_sol.g)
print(vns_sol.visited)
print("--- %s seconds ---" % (time.time() - start_time))

# test 2  --------------  OPT. SOL. = 30
places = [0, 1, 4, 9, 20, 18, 16, 5, 13, 19]
sol = initial_sol(montreal_graph, places)
start_time = time.time()
vns_sol = vns(sol=sol, k_max=10, t_max=1)
print(vns_sol.g)
print(vns_sol.visited)
print("--- %s seconds ---" % (time.time() - start_time))

# test 3  --------------  OPT. SOL. = 26
places = [0, 2, 7, 13, 11, 16, 15, 7, 9, 8, 4]
sol = initial_sol(montreal_graph, places)
start_time = time.time()
vns_sol = vns(sol=sol, k_max=10, t_max=1)
print(vns_sol.g)
print(vns_sol.visited)
print("--- %s seconds ---" % (time.time() - start_time))

# test 4  --------------  OPT. SOL. = 40
places = [0, 2, 20, 3, 18, 12, 13, 5, 11, 16, 15, 4, 9, 14, 1]
sol = initial_sol(montreal_graph, places)
start_time = time.time()
vns_sol = vns(sol=sol, k_max=10, t_max=1)
print(vns_sol.g)
print(vns_sol.visited)
print("--- %s seconds ---" % (time.time() - start_time))
