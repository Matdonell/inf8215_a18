# ----------------------------------------------------------------
# Authors: Mathieu Kabore, Florence Gaborit and Reph D. Mombrun
# Date: 11/09/2018
# Last update: 12/09/2018
# INF8215 TP1
# ----------------------------------------------------------------

import copy
import heapq

from edge import Edge

# Ensure to support Python 2 and 3 by using the correct import
try:
    import queue as q
except ImportError:
    import Queue as q

# This counter is used as a workaround to create new indexes for the contracted nodes
# We can find a better way
counter_fix_me = 10000000000000


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
                result = minimum_spanning_arborescence(new_sol, edges)
                new_sol.h = get_total_cost(result, new_sol)

                heapq.heappush(T, new_sol)

    return best_solution


def minimum_spanning_arborescence(sol, edges):
    """
    Returns the cost to reach the vertices in the unvisited list
    """
    if not isinstance(sol, Solution):
        raise ValueError("The parameter should be a type of Solution")

    best_in_edge = {}
    removed_edge_bag = {}
    original_edges = {}
    root = sol.visited[-1]  # The root is the last visited node
    destination_node = sol.not_visited[-1]

    for v in sol.not_visited:
        incoming_edge_to_v = []
        heapq.heapify(incoming_edge_to_v)

        for tmp in edges:
            if tmp.to_v == v:
                heapq.heappush(incoming_edge_to_v, tmp)
        best_in_edge[v] = heapq.heappop(incoming_edge_to_v)

        cycle = get_cycle(best_in_edge)

        if cycle is None:
            return best_in_edge

        # build a new graph in which C is contracted into a single node
        new_node = gen_node_code()  # creating the new contracted node

        # Get all the nodes not in the cycle C
        new_sub_nodes = [n for n in sol.not_visited
                         if not [c for c in cycle
                                 if c.from_v == n or c.to_v == n]]
        new_sub_nodes.append(new_node)
        new_edges = []

        for e in cycle:  # edges in the cycle
            for c in cycle:
                if c.from_v == e.from_v or c.to_v == e.from_v:
                    start_node_in_cycle = True

            for c in cycle:
                if c.from_v == e.to_v or c.to_v == e.to_v:
                    end_node_in_cycle = True

            new_edge = None
            if not start_node_in_cycle and not end_node_in_cycle:
                new_edge = e

            elif start_node_in_cycle and not end_node_in_cycle:
                new_edge = Edge(new_node, e.to_v, e.weight)

            elif end_node_in_cycle and not start_node_in_cycle:
                new_edge = Edge(e.from_v, new_node, 0)

                # save the best incoming edge to the new contracted edge
                removed_edge_bag[(new_edge.from_v, new_edge.to_v)] = best_in_edge[e.to_v]

                # Update the contracted edge weight by removing the cost of the removed one
                new_edge.weight = e.weight - removed_edge_bag[(new_edge.from_v, new_edge.to_v)].weight

            # if e.to_v == destination_node:
            #     break

            # Save the original edge and add the new one
            original_edges[new_edge] = e
            new_edges.append(new_edge)

        sub_tree = minimum_spanning_arborescence(new_sub_nodes, new_edges, root)

        # After saving the removed edge, the remainging is a minimum arborescence
        path = []
        if not [e for e in cycle
                if not [k for k in removed_edge_bag[sub_tree[new_node]]
                        if e.from_v != k.from_v and e.to_v != k.to_v]]:
            path.append(e)

        path.append(original_edges[new_edge])
        return path

    return best_in_edge


def get_cycle(edges):
    """
    Get the first cycle from the list of edges
    :param edges:
    :return:
    """
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
    """
    Calculate the total cost.
    :param list_of_edges:
    :param sol:
    :return:
    """
    cost = 0
    for edge in list_of_edges:
        cost += list_of_edges[edge].weight + sol.graph[sol.visited[-1]][list_of_edges[edge].from_v]

    return cost


def gen_node_code():
    """
    This method return a global counter but is used as a workaround
    The counter is used as a workaround to create new indexes for the contracted nodes
    We can find a better way
    Fix me after please
    :return:
    """
    global counter_fix_me
    counter_fix_me -= 1

    return counter_fix_me
