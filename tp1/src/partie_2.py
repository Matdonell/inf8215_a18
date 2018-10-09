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


def extract_sub_graph(root):
    g = {}

    # Generate all the outgoing edges from the root
    for i in range(len(root.not_visited)):
        weight = root.graph[root.visited[-1]][root.not_visited[i]]
        g[root.visited[-1]] = {root.not_visited[i]: weight}

    # Generate all edges between the nodes except the root
    for i in range(len(root.not_visited)):
        for j in range(len(root.not_visited)):
            # Avoid creating an edge between the same node (self-cycle)
            # Get the weight
            weight = root.graph[root.not_visited[i]][root.not_visited[j]]
            g[root.not_visited[j]] = {root.not_visited[i]: weight}

    return g


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

    edges = []
    heapq.heapify(edges)

    g = {}

    # Generate all the outgoing edges from the root
    for i in range(len(root.not_visited)):
        weight = root.graph[root.visited[-1]][root.not_visited[i]]
        edge = Edge(root.visited[-1], root.not_visited[i], weight)
        heapq.heappush(edges, edge)

    # Generate all edges between the nodes except the root
    for i in range(len(root.not_visited)):
        for j in range(len(root.not_visited)):

            # Avoid creating an edge between the same node (self-cycle)
            if root.not_visited[i] != root.not_visited[j]:
                # Get the weight
                weight = root.graph[root.not_visited[i]][root.not_visited[j]]

                # Create the node and add it to the queue
                edge = Edge(root.not_visited[i], root.not_visited[j], weight)
                heapq.heappush(edges, edge)

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
                new_sol.h = minimum_spanning_arborescence(extract_sub_graph(new_sol),
                                                          new_sol.visited[-1],
                                                          new_sol.not_visited[-1])
                heapq.heappush(T, new_sol)

    return best_solution


def minimum_spanning_arborescence(graph, root, dest):
    """
    Returns the cost to reach the vertices in the unvisited list
    """
    # if not isinstance(sol, Solution):
    #     raise ValueError("The parameter should be a type of Solution")

    # root = sol.visited[-1]  # The root is the last visited node
    # pm = sol.not_visited[-1]  # the destination node

    # def mst(root, G):
    reversed_graph = reverse(graph)

    if root in reversed_graph:
        reversed_graph[root] = {}
    g = {}

    for n in reversed_graph:
        if len(reversed_graph[n]) == 0:
            continue
        minimum = math.inf

        s, d = None, None
        for e in reversed_graph[n]:
            if reversed_graph[n][e] < minimum:
                minimum = reversed_graph[n][e]
                s, d = n, e

        if d in g:
            g[d][s] = reversed_graph[s][d]
        else:
            g[d] = {s: reversed_graph[s][d]}

    cycles = []
    visited = set()
    for n in g:
        if n not in visited:
            cycle = get_cycle(n, g, visited)
            cycles.append(cycle)

    rg = reverse(g)
    for cycle in cycles:
        if root in cycle:
            continue
        merge_cycles(cycle, graph, reversed_graph, g, rg)

    cost = 0
    first = None
    for s in g:
        if first is None:
            first = s
        for r in g[s]:
            cost += graph[s][r]

    return cost


def get_cycle(n, g, visited=None, cycle=None):
    if visited is None:
        visited = set()

    if cycle is None:
        cycle = []

    visited.add(n)
    cycle += [n]

    if n not in g:
        return cycle

    for e in g[n]:
        if e not in visited:
            cycle = get_cycle(e, g, visited, cycle)

    return cycle


def merge_cycles(cycle, graph, reversed_graph, g, rg):
    all_in_edges = []
    min_internal = None
    min_internal_weight = math.inf

    # find minimal internal edge weight
    for n in cycle:
        reversed_n = reversed_graph.get(n)

        if reversed_n is not None:
            for e in reversed_n:
                if e in cycle:
                    if min_internal is None or reversed_graph[n][e] < min_internal_weight:
                        min_internal = (n, e)
                        min_internal_weight = reversed_graph[n][e]
                        continue
                else:
                    all_in_edges.append((n, e))

    # find the incoming edge with minimum modified cost
    min_external = None
    min_modified_weight = 0

    for s, t in all_in_edges:
        u, v = rg[s].popitem()
        rg[s][u] = v

        w = reversed_graph[s][t] - (v - min_internal_weight)

        if min_external is None or min_modified_weight > w:
            min_external = (s, t)
            min_modified_weight = w

    u, w = rg[min_external[0]].popitem()

    rem = (min_external[0], u)
    rg[min_external[0]].clear()

    if min_external[1] in rg:
        rg[min_external[1]][min_external[0]] = w
    else:
        rg[min_external[1]] = {min_external[0]: w}

    if rem[1] in g:
        if rem[0] in g[rem[1]]:
            del g[rem[1]][rem[0]]

    if min_external[1] in g:
        g[min_external[1]][min_external[0]] = w
    else:
        g[min_external[1]] = {min_external[0]: w}


def reverse(graph):
    r = {}
    for src in graph:
        for (dst, c) in graph[src].items():
            if dst in r:
                r[dst][src] = c
            else:
                r[dst] = {src: c}
    return r
