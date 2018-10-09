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

    def add(self, idx):
        """
        Adds the point in position idx of not_visited list to the solution
        """
        # Check if we are a valid index before processing the task
        if len(self.not_visited) <= idx:
            raise ValueError("The input index is out of range for the not visited list")

        # Return and remove the not visited element at the given index
        node_to_visit = self.not_visited.pop(idx)
        last_visited = self.visited[-1]

        # Update the cost for the current move
        self.g += self.graph[last_visited][node_to_visit]

        # Mark the attraction as visited
        self.visited.append(node_to_visit)


def initial_sol(graph, places_to_visit):
    """
    Return a complete initial solution
    """
    assert len(places_to_visit) > 2

    sol = Solution(places_to_visit, graph)

    while len(sol.not_visited) != 1:
        idx = randint(0, len(sol.not_visited) - 1)
        sol.add(idx)
    sol.add(0)

    return sol


def shaking(sol, k):
    """
    Returns a solution on the k-th neighborhood of sol
    """
    assert len(sol.visited) >= 4
    new_sol = copy.deepcopy(sol)  # Clone the current solution
    for i in range(k):
        m = len(sol.visited) - 1  # sol is supposed to be a complete solution
        random_i = randint(1, m - 1)  # Since indices start at 0 for the visited python list
        random_j = random_i
        while random_j == random_i:
            random_j = randint(1, m - 1)
        new_sol.swap(random_i, random_j)  # Swap 2 cities
    return new_sol


def local_search_2opt(sol):
    """
    Apply 2-opt local search over sol
    """
    n=len(sol.visited)
    min=sol.g
    opt_sol=copy.deepcopy(sol)
    for j in range(2, n):
        for i in range(1, j):
            L=sol.visited[i+1:j-1]
            L=L.reverse()
            new_visited=sol.visited[:i-1]+sol.visited[j]+L+sol.visited[i]+sol.visited[j+1:]
            new_sol=copy.deepcopy(sol)
            new_sol.visited=new_visited
            new_sol.g=0
            for k in range(n-1):
                new_sol.g+=sol.graph[new_visited[k]][new_visited[k+1]]
            if new_sol.g<min:
                opt_sol=copy.deepcopy(new_sol.g)
                min=new_sol.g
            #pour recreer une solution avec la bonne liste et recalculer le cout
            #verifier que ca marche ?
    return opt_sol


def alternative_local_search_2opt(sol, must_restart_as_soon_better_found=False, t_init=-1, t_max=60000):
    """
    Apply 2-opt local search over sol.
    If there was amelioration, then this function will search again.
    Therefore, it could impact the execution time of VNS t_max.
    Here, we introduce new parameters: t_init and duration [ms] to better control
    the execution time of VNS via this function.
    """
    assert len(sol.visited) > 4  # At least 5 cities in this project: p1, i, i', j, pm

    # If the init time was not passed then initialize t_init with a valid value
    if t_init < 0:
        t_init = current_time_in_ms()

    # The solution to return
    opt_sol = copy.deepcopy(sol)

    # Note:
    # If m = len(sol.visited) - 1 is the index of pm then:
    # last_index_of_j = m - 1,
    # last_index_of_i_prime = m - 2 and
    # last_index_of_i = m - 3
    m = len(sol.visited) - 1

    # To detect if an optimum local was reach
    there_is_amelioration = True
    while there_is_amelioration:

        # Terminate when there is no more amelioration
        there_is_amelioration = False

        # At the first iteration, i = 1, i.e. the city following p1
        i = 1

        # Brief analysis: There are at most m - 4 iterations for i
        # and for the first iteration, there are m - 4 iterations for j.
        # For the 2nd iteration of i, there are m - 4 - 1 iterations for j.
        # ...
        # For the last iteration of i, there is only 1 iteration of j.
        # This is the Gauss's sum :
        # n + n - 1 + n - 2 + ... + 1 = n (n + 1) / 2, with n = m - 4
        while i <= m - 3:
            j = i + 2
            while j <= m - 1:

                # Swap i' with j, if the swap is not better,
                # then reverse it
                i_prime = i + 1
                current_cost = opt_sol.g
                opt_sol.swap(i_prime, j)
                if current_cost <= opt_sol.g:
                    opt_sol.swap(j, i_prime)
                else:
                    there_is_amelioration = True

                    # It t_max is reached, then return the current solution
                    if current_time_in_ms() - t_init >= t_max:
                        return opt_sol

                    # If the search should restart as soon a better solution is found then
                    # just do a recursive call with the current solution as argument
                    if must_restart_as_soon_better_found:
                        return alternative_local_search_2opt(opt_sol, True, t_init, t_max)

                j += 1
            i += 1

    return opt_sol


def vns(sol, k_max, t_max):
    """
    Performs the VNS algorithm
    """

    # Reminder: neighborhood k corresponds to the permutations
    # of k pairs of vertices

    t_init = current_time_in_ms()
    best_sol = sol
    duration = 0
    k = 1
    while (k <= k_max) and duration < t_max:

        # Generate a solution in the kth neighborhood
        new_sol = shaking(best_sol, k)

        # Do a local search on the solution generated from the kth neighborhood
        # new_sol = local_search_2opt(new_sol)  # FIXME: Error: can not do new_sol.g [line 226]
        new_sol = alternative_local_search_2opt(new_sol, False, t_init, t_max)

        # If the new solution has a better cost then,
        # update the current best solution
        if new_sol.g < best_sol.g:
            best_sol = copy.deepcopy(new_sol)  # FIXME : is deepcopy required here?

        # Keep track of the elapsed time and the neighborhood
        duration = current_time_in_ms() - t_init
        k += 1

    return best_sol


def current_time_in_ms():
    """
    Simply return the current time in ms
    :return: The current time in ms.
    """
    return int(round(time.time() * 1000))
