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
        self.graph = graph  # the adjacent matrix
        self.visited = [places[0]]  # list of already visited attractions
        self.not_visited = copy.deepcopy(places[1:])  # list of attractions not yet visited

    def swap(self, i, j):
        """
        :param i: node to swap with the j node
        :param j: node to swap with the i node
        :return: this solution
        """

        # The quantity of places to visit should be at least 4 for this project
        assert len(self.visited) > 3

        # The value of the i parameter should be in [1; len(self.visited) - 2]
        assert len(self.visited) - 1 > i > 0

        # The value of the j parameter should be in [1; len(self.visited) - 2]
        assert len(self.visited) - 1 > j > 0

        # Note : Another way to proceed could be to not do any swapping and just
        # return the original solution ...

        # Remove the cost of the following directions from self.g:
        # node[i-1] --> node[i] and node[i] --> node[i+1]
        # node[j-1] --> node[j] and node[j] --> node[j+1]
        cost = self.g
        cost = cost - self.graph[i - 1][i]
        cost = cost - self.graph[i][i + 1]
        cost = cost - self.graph[j - 1][j]
        cost = cost - self.graph[j][j + 1]

        # Do the swapping
        self.visited[i], self.visited[j] = self.visited[j], self.visited[i]

        # Add the new costs of the new directions
        cost = cost + self.graph[i - 1][i]
        cost = cost + self.graph[i][i + 1]
        cost = cost + self.graph[j - 1][j]
        cost = cost + self.graph[j][j + 1]
        self.g = cost

        return self

    def add(self, idx):
        """
        Adds the point in position idx of not_visited list to the solution
        """
        # Check if we are a valid index before processing the task
        if len(self.not_visited) <= idx:
            raise ValueError("The parameter should be a type of Solution")

        # Return and remove the not visited element at the given index
        node_to_visit = self.not_visited.pop(idx)
        last_visited = self.visited[-1]

        # Update the cost for the current move
        self.g += self.graph[last_visited][node_to_visit]

        # Mark the attraction as visited
        self.visited.append(node_to_visit)


def initial_sol(graph, places_to_visit):
    """
    Return a completed initial solution
    """
    assert len(places_to_visit) > 2

    sol = Solution(places_to_visit, graph)
    while len(sol.not_visited) != 1:
        idx = randint(0, len(sol.not_visited) - 1)
        sol = sol.add(idx)

    sol = sol.add(0)

    return sol


def shaking(sol, k):
    """
    Returns a solution on the k-th neighborhood of sol
    """

    assert(isinstance(sol, Solution))
    new_sol = copy.deepcopy(sol)  # Clone the current solution
    for i in range(k):
        m = len(sol.visited) - 1  # sol is supposed to be a complete solution
        random_i = randint(1, m - 2)  # since indices start at 0 for the visited python list
        random_j = random_i
        while random_j == random_i:
            random_j = randint(1, m - 2)
        new_sol = new_sol.swap(random_i, random_j) # Swap 2 cities
    return new_sol


def local_search_2opt(sol):
    """
    Apply 2-opt local search over sol
    """
    n=len(sol.visited)
    for j in range(2, n):
        for i in range(1, j):
            L=sol.visited[i+1:j-1]
            L=L.reverse()
            new_visited=sol.visited[:i-1]+sol.visited[j]+L+sol.visited[i]+sol.visited[j+1:]
            #recreer une solution avec la bonne liste et recalculer le cout + verifier que ca marche
    return sol


def vns(sol, k_max, t_max):
    """
    Performs the VNS algorithm
    """

    # hypothesis: neighborhood k corresponds to the permutations
    # of k pairs of vertices. Also, we supposed that the sol parameter
    # is the initial solution.

    t_init = current_time_in_ms()
    best_sol = sol
    duration = 0
    k = 1
    while (k <= k_max) and duration < t_max:

        # TODO : Generate a random solution in the kth neighborhood
        # ...

        # Do a local search in the kth neighborhood starting
        # from
        kth_neighborhood_best_sol = shaking(best_sol, k)
        if kth_neighborhood_best_sol.g < best_sol.g: # reminder: the lower the g cost is, better is the solution
            best_sol = kth_neighborhood_best_sol
        duration = current_time_in_ms() - t_init
        k += 1

    return best_sol


def current_time_in_ms():
    return int(round(time.time() * 1000))


def get_random_not_visited_adjacent(vertex, graph, visited):
    adjacents = []
    return None
