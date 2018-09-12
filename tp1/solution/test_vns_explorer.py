import unittest
import time

import partie_3 as VnsExplorer
from utility import Utility


class VnsExplorerTest(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super(VnsExplorerTest, self).__init__(methodName)

    def setUp(self):
        self.montreal_graph = Utility.read_graph()

    def test_1(self):
        # test 1  --------------  OPT. SOL. = 27
        expected_cost = 27

        places = [0, 5, 13, 16, 6, 9, 4]
        sol = VnsExplorer.initial_sol(self.montreal_graph, places)

        start_time = time.time()
        vns_sol = VnsExplorer.vns(sol=sol, k_max=10, t_max=1)

        print(vns_sol.g)
        print(vns_sol.visited)
        print("--- %s seconds ---" % (time.time() - start_time))
        self.assertEquals(vns_sol.g, expected_cost)

    def test_2(self):
        # test 2  --------------  OPT. SOL. = 30
        expected_cost = 30

        places = [0, 1, 4, 9, 20, 18, 16, 5, 13, 19]
        sol = VnsExplorer.initial_sol(self.montreal_graph, places)

        start_time = time.time()
        vns_sol = VnsExplorer.vns(sol=sol, k_max=10, t_max=1)

        print(vns_sol.g)
        print(vns_sol.visited)
        print("--- %s seconds ---" % (time.time() - start_time))

        self.assertEquals(vns_sol.g, expected_cost)

    def test_3(self):
        # test 3  --------------  OPT. SOL. = 26
        expected_cost = 26

        places = [0, 2, 7, 13, 11, 16, 15, 7, 9, 8, 4]
        sol = VnsExplorer.initial_sol(self.montreal_graph, places)

        start_time = time.time()
        vns_sol = VnsExplorer.vns(sol=sol, k_max=10, t_max=1)

        print(vns_sol.g)
        print(vns_sol.visited)
        print("--- %s seconds ---" % (time.time() - start_time))
        self.assertEquals(vns_sol.g, expected_cost)

    def test_4(self):
        # test 4  --------------  OPT. SOL. = 40
        expected_cost = 40

        places = [0, 2, 20, 3, 18, 12, 13, 5, 11, 16, 15, 4, 9, 14, 1]
        sol = VnsExplorer.initial_sol(self.montreal_graph, places)

        start_time = time.time()
        vns_sol = VnsExplorer.vns(sol=sol, k_max=10, t_max=1)

        print(vns_sol.g)
        print(vns_sol.visited)
        print("--- %s seconds ---" % (time.time() - start_time))
        self.assertEquals(vns_sol.g, expected_cost)


if __name__ == '__main__':
    unittest.main()
