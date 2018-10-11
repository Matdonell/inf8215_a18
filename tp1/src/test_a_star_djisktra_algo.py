# ----------------------------------------------------------------
# Authors: Mathieu Kabore, Florence Gaborit and Reph D. Mombrun
# Date: 12/09/2018
# Last update: 12/09/2018
# INF8215 TP1
# ----------------------------------------------------------------

import unittest
import time

import partie_2_1 as AStarExplorer
from utility import Utility


class AStarExplorerTest(unittest.TestCase):
    def __init__(self, method_name='runTest'):
        super(AStarExplorerTest, self).__init__(method_name)

    def setUp(self):
        self.montreal_graph = Utility.read_graph()

    def test_1(self):
        # test 1  --------------  OPT. SOL. = 27

        # Arrange
        places = [0, 5, 13, 16, 6, 9, 4]
        expected_cost = 27

        # Act
        start_time = time.time()
        astar_sol = AStarExplorer.A_star(graph=self.montreal_graph, places=places)
       
        print(astar_sol.g)
        print(astar_sol.visited)
        print("--- %s seconds ---" % (time.time() - start_time))

        # Assert
        self.assertEqual(expected_cost, astar_sol.g)

    def test_2(self):
        # test 2  --------------  OPT. SOL. = 30

        # Arrange
        places = [0, 1, 4, 9, 20, 18, 16, 5, 13, 19]
        expected_cost = 30

        # Act
        start_time = time.time()
        astar_sol = AStarExplorer.A_star(graph=self.montreal_graph, places=places)

        print(astar_sol.g)
        print(astar_sol.visited)
        print("--- %s seconds ---" % (time.time() - start_time))

        # Assert
        self.assertEqual(expected_cost, astar_sol.g)
    #
    def test_3(self):
        # test 3  --------------  OPT. SOL. = 26

        # Arrange
        places = [0, 2, 7, 13, 11, 16, 15, 7, 9, 8, 4]
        expected_cost = 26

        # Act
        start_time = time.time()
        astar_sol = AStarExplorer.A_star(graph=self.montreal_graph, places=places)

        print(astar_sol.g)
        print(astar_sol.visited)
        print("--- %s seconds ---" % (time.time() - start_time))

        # Assert
        self.assertEqual(expected_cost, astar_sol.g)

    def test_4(self):
        # test 4  --------------  OPT. SOL. = 40

        # Arrange
        places = [0, 2, 20, 3, 18, 12, 13, 5, 11, 16, 15, 4, 9, 14, 1]
        expected_cost = 40

        # Act
        start_time = time.time()
        astar_sol = AStarExplorer.A_star(graph=self.montreal_graph, places=places)

        print(astar_sol.g)
        print(astar_sol.visited)
        print("--- %s seconds ---" % (time.time() - start_time))

        # Assert
        self.assertEqual(expected_cost, astar_sol.g)


if __name__ == '__main__':
    unittest.main()
