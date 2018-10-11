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

infinit_val = 10000000000000


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

    edges = []
    heapq.heapify(edges)

    # Generate all edges between the nodes except the root
    for i in range(len(root.not_visited)):
        for j in range(len(root.not_visited)):

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

                new_edges = []
                for i in range(len(new_sol.not_visited)):
                    for j in range(len(new_sol.not_visited)):

                        if new_sol.not_visited[i] != new_sol.not_visited[j]:
                            # Get the weight
                            weight = new_sol.graph[new_sol.not_visited[i]][new_sol.not_visited[j]]

                            # Create the node and add it to the queue
                            edge = Edge(new_sol.not_visited[i], new_sol.not_visited[j], weight)

                            new_edges.append(edge)

                result = minimum_spanning_arborescence(new_sol, edges)

                new_sol.h = get_total_cost(result, new_sol)

                heapq.heappush(T, new_sol)

    return best_solution


def minimum_spanning_arborescence(sol, E):
    """
    Returns the cost to reach the vertices in the unvisited list
    """
    # if not isinstance(sol, Solution):
    #     raise ValueError("The parameter should be a type of Solution")

    # root = sol.visited[-1]  # The root is the last visited node
    # pm = sol.not_visited[-1]  # the destination node

    best_in_edge = {}
    removed_edge_bag = {}
    real = {}
    E = copy.deepcopy(E)

    V = sol.not_visited
    root = sol.visited[-1]

    for v in V:
        incoming_edge_to_v = []
        heapq.heapify(incoming_edge_to_v)

        for tmp in E:
            if tmp.to_v == v:
                heapq.heappush(incoming_edge_to_v, tmp)
        best_in_edge[v] = heapq.heappop(incoming_edge_to_v)

        cycle = get_cycle(best_in_edge)

        if cycle is None:
            return best_in_edge

        # build a new graph in which C is contracted into a single node
        new_node = gen_node_code()  # vC ← new Node
        new_sub_node = [n for n in V if not [c for c in cycle if c.from_v == n or c.to_v == n]]  # V0 ← V ∪ {vC } \ C
        new_sub_node.append(new_node)
        new_edges = []

        for e in cycle:  # e = (t, u) in E:
            contains_t = [c for c in cycle if c.from_v == e.from_v or c.to_v == e.from_v]
            contains_u = [c for c in cycle if c.from_v == e.to_v or c.to_v == e.to_v]

            new_edge = None
            if not contains_t and not contains_u:  # if t not ∈ C and u not∈ C:
                new_edge = e  # e0 ← e

            elif contains_t and not contains_u:
                new_edge = Edge(new_node, e.to_v, e.weight)  # e0 ← new Edge(vC, u)

            elif contains_u and not contains_t:
                new_edge = Edge(e.from_v, new_node, 0)

                # kicksOut[e0] ← bestInEdge[u]
                removed_edge_bag[(new_edge.from_v, new_edge.to_v)] = best_in_edge[e.to_v]

                # score[e0] ← score[e] − score[kicksOut[e0].weight]
                new_edge.weight = e.weight - removed_edge_bag[(new_edge.from_v, new_edge.to_v)].weight

            real[new_edge] = e  # remember the original edge
            new_edges.append(new_edge)  # E0 ← E0 ∪ {e0}

        sub_arborescence = minimum_spanning_arborescence(new_sub_node, new_edges, root)

        path = []
        if not [e for e in cycle if
                not [k for k in removed_edge_bag[sub_arborescence[new_node]]
                     if e.from_v != k.from_v and e.to_v != k.to_v]]:
            path.append(e)

        path.append(real[new_edge])
        return path  # return {real[e0] | e0 ∈ sub_arborescence} ∪ (CE \ {kicksOut[sub_arborescence[vC]]})

    return best_in_edge


def get_cycle(edges):
    visited = set()
    path = []
    path_set = set(path)
    stack = [iter(edges)]

    while stack:
        for v in stack[-1]:
            if v in path_set:
                return path_set
            elif not [e for e in visited if e.from_v == v.from_v and e.to_v == v.to_v]:
                visited.add(v)
                path.append(v)
                path_set.add(v)
                for e in edges:
                    if edges[e].from_v == v:
                        stack.append()
                        break
        else:
            if path:
                path_set.remove(path.pop())
            stack.pop()
    return None


def get_total_cost(list_of_edges, sol):
    cost = 0
    for edge in list_of_edges:
        cost += list_of_edges[edge].weight + sol.graph[sol.visited[-1]][list_of_edges[edge].from_v]
    return cost


def gen_node_code():
    global infinit_val
    infinit_val -= 1

    return infinit_val
