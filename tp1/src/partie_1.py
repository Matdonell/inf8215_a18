# ----------------------------------------------------------------
# Authors: Mathieu Kabore, Florence Gaborit and Reph D. Mombrun
# Date: 11/09/2018
# Last update: 12/09/2018
# INF8215 TP1
# ----------------------------------------------------------------

import numpy as np
import copy
import time
import math


# try:
#     import Queue as q
# except ImportError:
#     import q

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
        self.places = places

    def add(self, idx):
        s= copy.deepcopy(self)
       # print("id ", idx, "--", len(self.not_visited))
        s.g +=self.graph[s.visited[-1]][self.not_visited[idx]]
        s.visited.append(self.not_visited[idx])
        del s.not_visited[idx]
        return s
       
        """
        Adds the point in position idx of not_visited list to the solution
        """


def bfs(graph, places):
    
    """
    Returns the best solution which spans over all attractions indicated in 'places'
    """
    racine=Solution(places, graph) # premier noeud'
    q0 = [racine]  # création de la file initiale

    while len(q0[0].not_visited) >= 2: # tant que il reste des points non visités dans la première solution de la file'
         solution_mere=q0[0] # on prend le premier de la file'

         for i in range(len(solution_mere.not_visited)-1): # on construit ses enfants'
             #print("--", i)
             q0.append(solution_mere.add(i))

             #print ("before", q0)

         del q0[0] # on supprime le premier de la file'
        # print ("after", q0)


    min=math.inf
    optimum=q0[0]
    q1=[]

    for j in range(len(q0)):
        q1.append(q0[j].add(0))
       # print ("before2", q1) # ajout de pm'

        if q1[j].g<=min:
            min=q1[j].g
            optimum=q1[j]
           # print(min)

    return optimum
