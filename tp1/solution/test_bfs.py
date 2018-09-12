import unittest
import time

import partie_1 as BfsSearch
from utility import Utility


class BreadthFirstSearchTest(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super(BreadthFirstSearchTest, self).__init__(methodName)

    def setUp(self):
        self.montreal_graph = Utility.read_graph()

    def test_1(self):
        # test 1  --------------  OPT. SOL. = 27
        start_time = time.time()

        places = [0, 5, 13, 16, 6, 9, 4]
        sol = BfsSearch.bfs(graph=self.montreal_graph, places=places)

        print(sol.g)
        print("--- %s seconds ---" % (time.time() - start_time))

    def test_2(self):
        # test 2 -------------- OPT. SOL. = 30
        start_time = time.time()
        places = [0, 1, 4, 9, 20, 18, 16, 5, 13, 19]
        sol = BfsSearch.bfs(graph=self.montreal_graph, places=places)
        print(sol.g)
        print("--- %s seconds ---" % (time.time() - start_time))

    def test_3(self):
        # test 3 -------------- OPT. SOL. = 26
        start_time = time.time()
        places = [0, 2, 7, 13, 11, 16, 15, 7, 9, 8, 4]
        sol = BfsSearch.bfs(graph=self.montreal_graph, places=places)
        print(sol.g)
        print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    unittest.main()
