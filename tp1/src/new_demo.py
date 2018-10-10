# # ----------------------------------------------------------------
# # Authors: Mathieu Kabore, Florence Gaborit and Reph D. Mombrun
# # Date: 11/09/2018
# # Last update: 12/09/2018
# # INF8215 TP1
# # ----------------------------------------------------------------
#
# import copy
# import heapq
# import math
#
# from edge import Edge
#
# # Ensure to support Python 2 and 3 by using the correct import
# try:
#     import queue as q
# except ImportError:
#     import Queue as q
#
#
# class Solution:
#     def __init__(self, places, graph):
#         """
#         places: a list containing the indices of attractions to visit
#         p1 = places[0]
#         pm = places[-1]
#         """
#         self.g = 0  # current cost
#         self.h = 0  # the cost to go to the destination
#         self.graph = graph
#         self.visited = [places[0]]  # list of already visited attractions
#         self.not_visited = copy.deepcopy(places[1:])  # list of attractions not yet visited
#
#     def add(self, idx):
#         """
#         Adds the point in position idx of not_visited list to the solution
#         """
#         # Check if we are a valid index before processing the task
#         if len(self.not_visited) <= idx:
#             raise ValueError("The parameter should be a type of Solution")
#
#         # Return and remove the not visited element at the given index
#         node_to_visit = self.not_visited.pop(idx)
#         last_visited = self.visited[-1]
#
#         # Update the cost for the current move
#         self.g += self.graph[last_visited][node_to_visit]
#
#         # Mark the attraction as visited
#         self.visited.append(node_to_visit)
#
#     def __lt__(self, other):
#         if not isinstance(other, Solution):
#             raise ValueError("The parameter should be a type of Solution")
#         return (self.g + self.h) < (other.g + other.h)
#
#
# def fastest_path_estimation(sol):
#     """
#     Returns the time spent on the fastest path between
#     the current vertex c and the ending vertex pm
#     """
#     if not isinstance(sol, Solution):
#         raise ValueError("The parameter should be a type of Solution")
#
#     c = sol.visited[-1]  # the source node
#     pm = sol.not_visited[-1]  # the destination node
#
#     # create the search tree
#     T = []
#     heapq.heapify(T)
#
#     distance_map = {}
#
#     # Initialization
#     for i in range(len(sol.not_visited)):
#         sub_node = copy.deepcopy(sol)
#         sub_node.add(i)
#         sub_node.g = math.inf
#
#         # init all distances with infinite value for the sub nodes
#         distance_map[sol.not_visited[i]] = math.inf
#         heapq.heappush(T, sub_node)
#
#     heapq.heappush(T, sol)
#     distance_map[c] = 0
#
#     while T:
#
#         # Return and remove the node with the best cost
#         best_sol = heapq.heappop(T)
#
#         # If the last visited is the destination, we are done
#         if best_sol.visited[-1] == pm:
#             return best_sol.g
#
#         # For each unvisited sub node of the current node
#         for i in range(len(best_sol.not_visited)):
#             last_visited = best_sol.visited[-1]
#
#             # calculate the cost of the current move
#             new_cost = distance_map[last_visited] + sol.graph[last_visited, best_sol.not_visited[i]]
#
#             # If we have a best move with less cost, let's update the new cost
#             if new_cost < distance_map[best_sol.not_visited[i]]:
#                 distance_map[best_sol.not_visited[i]] = new_cost
#
#                 sub_node = copy.deepcopy(best_sol)
#                 sub_node.add(i)
#                 sub_node.g = new_cost
#                 heapq.heappush(T, sub_node)
#
#     return None
#
#
# def A_star(graph, places):
#     """
#     Performs the A* algorithm
#     """
#
#     # blank solution
#     root = Solution(graph=graph, places=places)
#
#     # search tree T
#     T = []
#     heapq.heapify(T)
#     heapq.heappush(T, root)
#     best_solution = copy.deepcopy(root)
#
#     edges = []
#     heapq.heapify(edges)
#
#     # Generate all the outgoing edges from the root
#     for i in range(len(root.not_visited)):
#         weight = root.graph[root.visited[-1]][root.not_visited[i]]
#         edge = Edge(root.visited[-1], root.not_visited[i], weight)
#         heapq.heappush(edges, edge)
#
#     # Generate all edges between the nodes except the root
#     for i in range(len(root.not_visited)):
#         for j in range(len(root.not_visited)):
#
#             # Avoid creating an edge between the same node (self-cycle)
#             if root.not_visited[i] != root.not_visited[j]:
#                 # Get the weight
#                 weight = root.graph[root.not_visited[i]][root.not_visited[j]]
#
#                 # Create the node and add it to the queue
#                 edge = Edge(root.not_visited[i], root.not_visited[j], weight)
#
#                 heapq.heappush(edges, edge)
#
#     # new_sol = minimum_spanning_arborescence(root.not_visited, edges, root.visited[-1], root.not_visited[-1])
#
#     # print("total_cost", new_sol)
#     #
#     while best_solution.not_visited:
#         best_solution = heapq.heappop(T)
#
#         # Since we are skipping the destination node pm when expanding the sub nodes
#         # if we have one last node to visit for a specific branch (pm)
#         # Add it into the solution and stop the search
#         if len(best_solution.not_visited) == 1:
#             best_solution.add(0)
#             return best_solution
#         else:
#             # Generate the sub nodes (a.k.a expand the sub solutions) from the current
#             # Notice: skip the destination attraction for now
#             # And find the fastest path between the node and the destination
#             for i in range(len(best_solution.not_visited) - 1):
#                 new_sol = copy.deepcopy(best_solution)
#                 new_sol.add(i)
#
#                 # Update the fastest path to pm
#                 # new_sol.h = fastest_path_estimation(new_sol)
#                 new_sol.h = get_total_cost(minimum_spanning_arborescence(root.not_visited, edges, root.visited[-1]), root.not_visited[-1])
#                 heapq.heappush(T, new_sol)
#
#     return best_solution
#
#
# def minimum_spanning_arborescence(V, edges, root):
#     """
#     Returns the cost to reach the vertices in the unvisited list
#     """
#     # if not isinstance(sol, Solution):
#     #     raise ValueError("The parameter should be a type of Solution")
#
#     # root = sol.visited[-1]  # The root is the last visited node
#     # pm = sol.not_visited[-1]  # the destination node
#
#     # new_node_value = 100
#     #
#     # best_in_edge = {}
#     # kicks_out = {}
#     # real = {}
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
#     # print("------")
#     # for edge in edges:
#     #     edge.print()
#
#     P = []
#     # For each node v != root
#     for node in V:
#         # P = Find all incoming source t to v with lowest cost and != root
#         P = [edge for edge in edges if edge.to_v == node and edge.from_v != root]
#
#         # for edge in P:
#         #     edge.print()
#
#         # if P does not contains a cycle:
#         # return P Research DONE!
#
#         for edge in P:
#             cycle = [edge]
#             tmp_edges = [e for e in P if e.from_v != edge.from_v or e.to_v != edge.to_v]
#             cycle = get_cycle(edge, tmp_edges, cycle)
#             if len(cycle) > 2:
#                 break
#
#         if cycle is None or len(cycle) < 3:
#             return P
#
#         # TODO: Calculate the cost
#         # print("print cycle: ")
#         # for c in cycle:
#         #     c.print()
#         # print("end of cycle: ")
#
#         # Else: P contains at least 1 cycle
#         # C = get a random cycle from P
#         # D' = Define a new weighted directed Graph D' in which C is contracted as follow:
#
#         # build a new graph in which C is contracted into a single node
#         v_C = new_node_value  # vC ← new Node
#         V_0 = [n for n in V if not [c for c in cycle if c.from_v == n or c.to_v == n]]  # V0 ← V ∪ {vC } \ C
#         V_0.append(v_C)
#         E_0 = []
#
#         for e in edges:  # e = (t, u) in E:
#             contains_t = [c for c in cycle if c.from_v == e.from_v or c.to_v == e.from_v]
#             contains_u = [c for c in cycle if c.from_v == e.to_v or c.to_v == e.to_v]
#
#             if not contains_t and not contains_u:  # if t not ∈ C and u not∈ C:
#                 e_0 = e  # e0 ← e
#
#             elif contains_t and not contains_u:
#                 e_0 = Edge(v_C, e.to_v, e.weight)  # e0 ← new Edge(vC, u)
#                 # score[e0] ← score[e]
#
#             elif contains_u and not contains_t:
#                 # e0 ← new Edge(t, vC)
#                 e_0 = Edge(e.from_v, v_C, 0)
#
#                 # kicksOut[e0] ← bestInEdge[u]
#                 kicks_out[(e_0.from_v, e_0.to_v)] = best_in_edge[e.to_v]
#
#                 # score[e0] ← score[e] − score[kicksOut[e0].weight]
#                 e_0.weight = e.weight - kicks_out[(e_0.from_v, e_0.to_v)].weight
#
#             real[e_0] = e  # remember the original
#             E_0.append(e_0)  # E0 ← E0 ∪ {e0}
#
#         A = minimum_spanning_arborescence(V_0, E_0, root)  # A ← Get1Best({V0, E0}, ROOT )
#
#         path = []
#         if len(A) >= v_C:
#             for e in cycle:
#                 if not [k for k in kicks_out[A[v_C]] if e.from_v != k.from_v and e.to_v != k.to_v]:
#                     path.append(e)
#
#         path.append(real[e_0])
#         return path  # return {real[e0] | e0 ∈ A} ∪ (CE \ {kicksOut[A[vC]]})
#
#     #     # Call recursivelly minimum_spanning_arborescence(D')
#     #
#     # for edge in P:
#     #     lowest_cost += edge.weight
#
#     # *************************************
#
#     # for v in V:
#     #     incoming_edge_to_v = []
#     #     heapq.heapify(incoming_edge_to_v)
#     #
#     #     for tmp in E:
#     #         if tmp.to_v == v:
#     #             heapq.heappush(incoming_edge_to_v, tmp)
#     #     best_in_edge[v] = heapq.heappop(incoming_edge_to_v)
#     #
#     #     for e in best_in_edge:
#     #         cycle = get_cycle(e, copy.deepcopy(E))
#     #         if cycle is not None:
#     #             break
#     #
#     #     if cycle is None:
#     #         return best_in_edge
#     #
#     #     # build a new graph in which C is contracted into a single node
#     #     v_C = new_node_value  # vC ← new Node
#     #     V_0 = [n for n in V if not [c for c in cycle if c.from_v == n or c.to_v == n]]  # V0 ← V ∪ {vC } \ C
#     #     V_0.append(v_C)
#     #     E_0 = []
#     #
#     #     for e in E:  # e = (t, u) in E:
#     #         contains_t = [c for c in cycle if c.from_v == e.from_v or c.to_v == e.from_v]
#     #         contains_u = [c for c in cycle if c.from_v == e.to_v or c.to_v == e.to_v]
#     #
#     #         if not contains_t and not contains_u:  # if t not ∈ C and u not∈ C:
#     #             e_0 = e  # e0 ← e
#     #
#     #         elif contains_t and not contains_u:
#     #             e_0 = Edge(v_C, e.to_v, e.weight)  # e0 ← new Edge(vC, u)
#     #             # score[e0] ← score[e]
#     #
#     #         elif contains_u and not contains_t:
#     #             # e0 ← new Edge(t, vC)
#     #             e_0 = Edge(e.from_v, v_C, 0)
#     #
#     #             # kicksOut[e0] ← bestInEdge[u]
#     #             kicks_out[(e_0.from_v, e_0.to_v)] = best_in_edge[e.to_v]
#     #
#     #             # score[e0] ← score[e] − score[kicksOut[e0].weight]
#     #             e_0.weight = e.weight - kicks_out[(e_0.from_v, e_0.to_v)].weight
#     #
#     #         real[e_0] = e  # remember the original
#     #         E_0.append(e_0)  # E0 ← E0 ∪ {e0}
#     #
#     #     A = minimum_spanning_arborescence(V_0, E_0, root)  # A ← Get1Best({V0, E0}, ROOT )
#     #
#     #     path = []
#     #     if len(A) >= v_C:
#     #         for e in cycle:
#     #             if not [k for k in kicks_out[A[v_C]] if e.from_v != k.from_v and e.to_v != k.to_v]:
#     #                 path.append(e)
#     #
#     #     path.append(real[e_0])
#     #     return path  # return {real[e0] | e0 ∈ A} ∪ (CE \ {kicksOut[A[vC]]})
#
#     return best_in_edge
#
#
# def get_cycle(edge, edges_list, cycle=None):
#     """ best_in_edge : list of edge """
#
#     incidence_matrix = [e for e in edges_list if e.from_v == edge.to_v]
#
#     if not incidence_matrix:
#         return cycle
#
#     first_incident = incidence_matrix[0]
#
#     if cycle is None:
#         cycle = []
#
#     if cycle[0].from_v == first_incident.to_v:
#         cycle.append(first_incident)
#         return cycle
#
#     edges_list.remove(first_incident)
#     cycle = get_cycle(first_incident, edges_list, cycle)
#
#     # print('start cycle')
#     # for c in cycle:
#     #     c.print()
#     # print('end cycle')
#
#     return cycle
#
#
# def get_total_cost(list_of_edges, dest):
#     cost = 0
#     for edge in list_of_edges:
#         cost += edge.weight
#         if edge.from_v == dest or edge.to_v == dest:
#             return cost
#     return cost
