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

    edges = []
    heapq.heapify(edges)

    # Generate all the outgoing edges from the root
    # for i in range(len(root.not_visited)):
    #     weight = root.graph[root.visited[-1]][root.not_visited[i]]
    #     edge = Edge(root.visited[-1], root.not_visited[i], weight)
    #     heapq.heappush(edges, edge)

    # We first remove any edge from E whose destination is r
    # Generate all edges between the nodes except the root

    graph = {}
    vertex = set()
    for i in range(len(places)):
        for j in range(len(places)):

            if places[i] != places[j]:
                # Get the weight
                weight = root.graph[places[i]][places[j]]

                # Create the node and add it to the queue
                edge = Edge(places[i], places[j], weight)

                # the graph
                graph[places[i]] = {places[j]: weight}

                heapq.heappush(edges, edge)

    # We may also replace any set of parallel edges (edges between the same pair
    #  of vertices in the same direction) by a single edge with weight equal to
    #  the minimum of the weights of these parallel edges.

    # for edge in edges:
    #     edge.print()
    # for e in edges:
    #     for ed in edges:
    #         if e.from_v != ed.from_v and e.to_v != ed.to_v and e.from_v == ed.to_v and e.to_v == ed.from_v:
    #             if e.weight <= ed.weight:
    #                 edges.remove(ed)
    #             else:
    #                 edges.remove(e)

    # print('\nafter')
    #
    # for edge in edges:
    #     edge.print()

    new_sol = minimum_spanning_arborescence(places, edges, root.visited[-1])

    print("result : ", new_sol)

    print("total_cost", get_total_cost(new_sol))

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
                new_sol.h = get_total_cost(minimum_spanning_arborescence(new_sol.not_visited, edges, new_sol.visited[-1]))
                heapq.heappush(T, new_sol)

    return best_solution


infinit_val = 10000000000000


def gen_node_code():
    global infinit_val
    infinit_val -= 1

    return infinit_val


def minimum_spanning_arborescence(vertexes, edges, root):
    """
    Returns the cost to reach the vertices in the unvisited list
    """
    #
    # print("\n deb", root)
    # for edge in edges:
    #     edge.print()
    #
    # print("\nfin", root)

    # We first remove any edge from E whose destination is r
    for e in edges:
        if e.to_v == root:
            edges.remove(e)

    # We may also replace any set of parallel edges (edges between the same pair
    #  of vertices in the same direction) by a single edge with weight equal to
    #  the minimum of the weights of these parallel edges.

    # print("\nno root", root)
    # for edge in edges:
    #     edge.print()
    #
    # print("\nfin no root", root)

    for e in edges:
        for ed in edges:
            if e.from_v != ed.from_v and e.to_v != ed.to_v and e.from_v == ed.to_v and e.to_v == ed.from_v:
                if e.weight <= ed.weight:
                    edges.remove(ed)
                else:
                    edges.remove(e)

    all_best_in = []  # P
    real = {}

    # for each node v other than the root,
    for v in vertexes:
        # find the edge incoming to v of lowest weight (with ties broken arbitrarily).

        edges_in_v = []
        heapq.heapify(edges_in_v)

        for e in edges:
            if e.to_v == v and v != root:
                heapq.heappush(edges_in_v, e)

        if edges_in_v:
            edge_best_in = heapq.heappop(edges_in_v)
            all_best_in.append(edge_best_in)

    # get a cycle from all_best_in = P
    cycle = get_cycle(all_best_in)  # Cycle C

    if not cycle:
        return all_best_in
        #
        # print('cycle: ')
        # for e in cycle:
        #     e.print()

        # The nodes of V^ are the nodes of  V not in C, plus a new node denoted new_node.
        # D' = (V', D')

    new_vertexes = [n for n in vertexes if not [c for c in cycle if c.from_v == n or c.to_v == n]]  # V0 ← V ∪ {vC } \ C
    new_edges = []

    for e in edges:  # e = (t, u) in E:
        contains_t = [c for c in cycle if c.from_v == e.from_v or c.to_v == e.from_v]
        contains_u = [c for c in cycle if c.from_v == e.to_v or c.to_v == e.to_v]

        new_node = gen_node_code()  # vC ← new Node
        new_vertexes.append(new_node)

        if not contains_t and contains_u:

            weight = e.weight - [n for n in cycle if n.to_v == e.to_v][0].weight
            new_edge = Edge(e.from_v, new_node, weight)

        elif contains_t and not contains_u:
            new_edge = Edge(new_node, e.to_v, e.weight)  # e0 ← new Edge(vC, u)

        elif not contains_t and not contains_u:  # if t not ∈ C and u not∈ C:
            new_edge = copy.deepcopy(e)  # e0 ← e

        real[new_edge] = copy.deepcopy(e)  # remember the original
        new_edges.append(new_edge)  # E0 ← E0 ∪ {e0}

    A = minimum_spanning_arborescence(new_vertexes, new_edges, root)  # A ← Get1Best({V0, E0}, ROOT )

    # path = []
    # if len(A) >= new_node:
    #     if not [e for e in cycle if
    #             not [k for k in kicks_out[A[new_node]] if e.from_v != k.from_v and e.to_v != k.to_v]]:
    #         path.append(e)
    #
    # path.append(real[new_edge])
    # return path  # return {real[e0] | e0 ∈ A} ∪ (CE \ {kicksOut[A[vC]]})
    A[new_node]

    #  Let (u,v_C) be the unique incoming edge to v_C in A'.
    #  This edge corresponds to an edge (u,v) in E with  v in C.
    #  Remove the edge (pi(v),v) from C, breaking the cycle.
    # Mark each remaining edge in C.
    #  For each edge in A', mark its corresponding edge in E.

    # Now we define f(D, r, w) to be the set of marked edges,
    # which form a minimum spanning arborescence.


    return None
    #
    # new_node_value = len(V)
    #
    # best_in_edge = {}
    # kicks_out = {}
    # real = {}
    # E = copy.deepcopy(E)
    #
    # for v in V:
    #     incoming_edge_to_v = []
    #     heapq.heapify(incoming_edge_to_v)
    #
    #     for tmp in E:
    #         if tmp.to_v == v:
    #             heapq.heappush(incoming_edge_to_v, tmp)
    #     best_in_edge[v] = heapq.heappop(incoming_edge_to_v)
    #
    #     for e in best_in_edge:
    #         cycle = get_cycle(e, copy.deepcopy(E))
    #         if cycle is not None:
    #             break
    #
    #     if cycle is None:
    #         return best_in_edge
    #
    #     # build a new graph in which C is contracted into a single node
    #     new_node = new_node_value  # vC ← new Node
    #     new_vertexes = [n for n in V if not [c for c in cycle if c.from_v == n or c.to_v == n]]  # V0 ← V ∪ {vC } \ C
    #     new_vertexes.append(new_node)
    #     new_edges = []
    #
    #     for e in E:  # e = (t, u) in E:
    #         contains_t = [c for c in cycle if c.from_v == e.from_v or c.to_v == e.from_v]
    #         contains_u = [c for c in cycle if c.from_v == e.to_v or c.to_v == e.to_v]
    #
    #         new_edge = None
    #         if not contains_t and not contains_u:  # if t not ∈ C and u not∈ C:
    #             new_edge = e  # e0 ← e
    #
    #         elif contains_t and not contains_u:
    #             new_edge = Edge(new_node, e.to_v, e.weight)  # e0 ← new Edge(vC, u)
    #             # score[e0] ← score[e]
    #
    #         elif contains_u and not contains_t:
    #             # e0 ← new Edge(t, vC)
    #             new_edge = Edge(e.from_v, new_node, 0)
    #
    #             # kicksOut[e0] ← bestInEdge[u]
    #             kicks_out[(new_edge.from_v, new_edge.to_v)] = best_in_edge[e.to_v]
    #
    #             # score[e0] ← score[e] − score[kicksOut[e0].weight]
    #             new_edge.weight = e.weight - kicks_out[(new_edge.from_v, new_edge.to_v)].weight
    #
    #         real[new_edge] = e  # remember the original
    #         new_edges.append(new_edge)  # E0 ← E0 ∪ {e0}
    #
    #     A = minimum_spanning_arborescence(new_vertexes, new_edges, root)  # A ← Get1Best({V0, E0}, ROOT )
    #
    #     path = []
    #     if len(A) >= new_node:
    #         if not [e for e in cycle if
    #                 not [k for k in kicks_out[A[new_node]] if e.from_v != k.from_v and e.to_v != k.to_v]]:
    #             path.append(e)
    #
    #     path.append(real[new_edge])
    #     return path  # return {real[e0] | e0 ∈ A} ∪ (CE \ {kicksOut[A[vC]]})

    return None


def get_total_cost(list_of_edges):
    cost = 0
    for edge in list_of_edges:
        cost += edge.weight

    return cost


# def minimum_spanning_arborescence(sol):
#     """
#     Returns the cost to reach the vertices in the unvisited list
#     """
#     if not isinstance(sol, Solution):
#         raise ValueError("The parameter should be a type of Solution")
#
#     root = sol.visited[-1]  # The root is the last visited node
#     pm = sol.not_visited[-1]  # the destination node
#     lowest_cost = 0
#
#     edges = []
#     heapq.heapify(edges)
#
#     for i in range(len(sol.not_visited)):
#         weight = sol.graph[root][sol.not_visited[i]]
#         edge = Edge(root, sol.not_visited[i], weight)
#         heapq.heappush(edges, edge)
#
#     for i in range(len(sol.not_visited)):
#         for j in range(len(sol.not_visited)):
#             weight = sol.graph[sol.not_visited[i]][sol.not_visited[j]]
#             edge = Edge(sol.not_visited[i], sol.not_visited[j], weight)
#             heapq.heappush(edges, edge)
#
#     # Remove all edges from E with destination 'root'
#     # Replace parallels edges with the minimum of them
#     for edge in edges:
#         for e in edges:
#             if e.from_v == edge.to_v and e.to_v == edge.from_v:
#                 if e.weight < edge.weight:
#                     edges.remove(edge)
#                 else:
#                     edges.remove(e)
#
#     P = []
#     # For each node v != root
#     for node in sol.not_visited:
#         # P = Find all incoming source t to v with lowest cost and != root
#         P = [edge for edge in edges if edge.to_v == node]
#
#         # for edge in P:
#         #     edge.print()
#
#         # if P does not contains a cycle:
#         # return P Research DONE!
#         # cycle = get_cycle(node, P)
#
#         # print(cycle)
#         return
#
#     #     if cycle is not None:
#     #         return
#     #
#     #     # Else: P contains at least 1 cycle
#     #     # C = get a random cycle from P
#     #     # D' = Define a new weighted directed Graph D' in which C is contracted as follow:
#     #     # Call recursivelly minimum_spanning_arborescence(D')
#     #
#     # for edge in P:
#     #     lowest_cost += edge.weight
#
#     return lowest_cost


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
                stack.append(iter([e for e in edges if e.from_v == v.to_v]))
                break
        else:
            if path:
                path_set.remove(path.pop())
            stack.pop()
    return None
