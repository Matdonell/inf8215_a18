# ----------------------------------------------------------------
# Authors: Mathieu Kabore, Florence Gaborit and Reph D. Mombrun
# Date: 11/09/2018
# Last update: 12/09/2018
# INF8215 TP1
# ----------------------------------------------------------------

import copy
import heapq
import math

from edge import Edge

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
        self.h = 0  # the cost to go to the destination
        self.graph = graph
        self.visited = [places[0]]  # list of already visited attractions
        self.not_visited = copy.deepcopy(places[1:])  # list of attractions not yet visited

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
        return (self.g + self.h) < (other.g + other.h)


def fastest_path_estimation(sol):
    """
    Returns the time spent on the fastest path between
    the current vertex c and the ending vertex pm
    """
    if not isinstance(sol, Solution):
        raise ValueError("The parameter should be a type of Solution")

    c = sol.visited[-1]  # the source node
    pm = sol.not_visited[-1]  # the destination node

    # create the search tree
    T = []
    heapq.heapify(T)

    distance_map = {}

    # Initialization
    for i in range(len(sol.not_visited)):
        sub_node = copy.deepcopy(sol)
        sub_node.add(i)
        sub_node.g = math.inf

        # init all distances with infinite value for the sub nodes
        distance_map[sol.not_visited[i]] = math.inf
        heapq.heappush(T, sub_node)

    heapq.heappush(T, sol)
    distance_map[c] = 0

    while T:

        # Return and remove the node with the best cost
        best_sol = heapq.heappop(T)

        # If the last visited is the destination, we are done
        if best_sol.visited[-1] == pm:
            return best_sol.g

        # For each unvisited sub node of the current node
        for i in range(len(best_sol.not_visited)):
            last_visited = best_sol.visited[-1]

            # calculate the cost of the current move
            new_cost = distance_map[last_visited] + sol.graph[last_visited, best_sol.not_visited[i]]

            # If we have a best move with less cost, let's update the new cost
            if new_cost < distance_map[best_sol.not_visited[i]]:
                distance_map[best_sol.not_visited[i]] = new_cost

                sub_node = copy.deepcopy(best_sol)
                sub_node.add(i)
                sub_node.g = new_cost
                heapq.heappush(T, sub_node)

    return None


def A_star(graph, places):
    """
    Performs the A* algorithm
    """

    # blank solution
    root = Solution(graph=graph, places=places)

    # search tree T
    T = []
    heapq.heapify(T)
    heapq.heappush(T, root)
    best_solution = copy.deepcopy(root)

    while best_solution.not_visited:
        best_solution = heapq.heappop(T)

        # Since we are skipping the destination node pm when expanding the sub nodes
        # if we have one last node to visit for a specific branch (pm)
        # Add it into the solution and stop the search
        if len(best_solution.not_visited) == 1:
            best_solution.add(0)
            return best_solution
        else:
            # Generate the sub nodes (a.k.a expand the sub solutions) from the current
            # Notice: skip the destination attraction for now
            # And find the fastest path between the node and the destination
            for i in range(len(best_solution.not_visited) - 1):
                new_sol = copy.deepcopy(best_solution)
                new_sol.add(i)

                # Update the fastest path to pm
                # new_sol.h = fastest_path_estimation(new_sol)
                new_sol.h = minimum_spanning_arborescence(new_sol)
                heapq.heappush(T, new_sol)

    return best_solution


def minimum_spanning_arborescence(sol):
    """
    Returns the cost to reach the vertices in the unvisited list
    """
    if not isinstance(sol, Solution):
        raise ValueError("The parameter should be a type of Solution")

    root = sol.visited[-1]  # The root is the last visited node
    pm = sol.not_visited[-1]  # the destination node
    lowest_cost = 0

    edges = []
    heapq.heapify(edges)

    for i in range(len(sol.not_visited)):
        weight = sol.graph[root][sol.not_visited[i]]
        edge = Edge(root, sol.not_visited[i], weight)
        heapq.heappush(edges, edge)

    for i in range(len(sol.not_visited)):
        for j in range(len(sol.not_visited)):
            if sol.not_visited[i] != sol.not_visited[j]:
                weight = sol.graph[sol.not_visited[i]][sol.not_visited[j]]
                edge = Edge(sol.not_visited[i], sol.not_visited[j], weight)
                heapq.heappush(edges, edge)

    # Remove all edges from E with destination 'root'
    # Replace parallels edges with the minimum of them
    for edge in edges:
        for e in edges:
            if e.from_v == edge.to_v and e.to_v == edge.from_v:
                if e.weight < edge.weight:
                    edges.remove(edge)
                else:
                    edges.remove(e)

    P = []
    # For each node v != root
    for node in sol.not_visited:
        # P = Find all incoming source t to v with lowest cost and != root
        P = [edge for edge in edges if edge.to_v == node]

        # for edge in P:
        #     edge.print()

        # if P does not contains a cycle:
        # return P Research DONE!
        # cycle = get_cycle(node, P)

        # print(cycle)
        return

    #     if cycle is not None:
    #         return
    #
    #     # Else: P contains at least 1 cycle
    #     # C = get a random cycle from P
    #     # D' = Define a new weighted directed Graph D' in which C is contracted as follow:
    #     # Call recursivelly minimum_spanning_arborescence(D')
    #
    # for edge in P:
    #     lowest_cost += edge.weight

    return lowest_cost
