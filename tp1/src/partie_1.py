# ----------------------------------------------------------------
# Authors: Mathieu Kabore, Florence Gaborit and Reph D. Mombrun
# Date: 11/09/2018
# Last update: 12/09/2018
# INF8215 TP1
# ----------------------------------------------------------------

import copy
import heapq

# Ensure to support Python 2 and 3 by using the correct import
try:
    import queue as q
except ImportError:
    import Queue as q


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

    def __lt__(self, other):
        if not isinstance(other, Solution):
            raise ValueError("The parameter should be a type of Solution")
        return self.g < other.g


def bfs(graph, places):
    """
    Returns the best solution which spans over all attractions indicated in 'places'
    """

    # Create the initial solution and store it in the partial solutions queue
    initial_node = Solution(places, graph)
    partial_sol = q.Queue()
    partial_sol.put(initial_node)

    # Create a priority queue to store all complete solutions (Predicate = cost = g)
    complete_sol = []
    heapq.heapify(complete_sol)

    # Generate all the possible solutions
    while not partial_sol.empty():

        # Return and remove the solution (node) in the bottom of the queue
        current_node = partial_sol.get()

        # Generate the sub nodes (a.k.a expand the sub solutions) from the current
        # Notice: skip the destination attraction for now
        for i in range(len(current_node.not_visited) - 1):
            sub_node = copy.deepcopy(current_node)
            sub_node.add(i)

            # At the last not visited attraction (pm),
            # add it and push the complete solution into the priority queue
            if len(sub_node.not_visited) == 1:
                sub_node.add(0)
                heapq.heappush(complete_sol, sub_node)
            else:
                # Adding the partial sub solution at this state to the partial solutions queue
                partial_sol.put(sub_node)

    # Return the complete solution with the lowest cost
    # Based on the predicate related to the cost defined in the Solution class __lt__
    return heapq.heappop(complete_sol)
