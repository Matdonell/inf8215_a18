# ----------------------------------------------------------------
# Authors: Mathieu Kaboré, Florence Gaborit and Reph D. Mombrun
# Date: 12/09/2018
# Last update: 02/11/2018
# INF8215 TP3
# ----------------------------------------------------------------

import unittest
import time
import numpy as np
from SoftmaxClassifier import SoftmaxClassifier


class SoftmaxClassifierTests(unittest.TestCase):
    def __init__(self, method_name='runTest'):
        super(SoftmaxClassifierTests, self).__init__(method_name)

    def setUp(self):
        self.softmaxClassifier = SoftmaxClassifier()

    def test_onehot_encoder(self):
        # Arrange
        y = [1, 1, 2, 3, 1]

        expected = np.array([[1, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0]]).tolist()
        # Act
        yohe = self.softmaxClassifier._one_hot(y)

        # Assert
        self.assertEqual(expected, yohe)

if __name__ == '__main__':
    unittest.main()
