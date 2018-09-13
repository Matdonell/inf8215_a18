# ----------------------------------------------------------------
# Authors: Mathieu Kabore, Florence Gaborit and Reph D. Mombrun
# Date: 11/09/2018
# Last update: 12/09/2018
# INF8215 TP1
# ----------------------------------------------------------------

import numpy as np


class Utility:
    """
    Represents an utility with useful fonctions.
    """
    @staticmethod
    def read_graph():
        return np.loadtxt("montreal", dtype='i', delimiter=',')
