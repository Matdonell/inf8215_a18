# ----------------------------------------------------------------
# Authors: Mathieu Kaboré, Florence Gaborit and Reph D. Mombrun
# Date: 12/09/2018
# Last update: 02/11/2018
# INF8215 TP3
# ----------------------------------------------------------------

import unittest
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from SoftmaxClassifier import SoftmaxClassifier


class SoftmaxClassifierTests(unittest.TestCase):
    def __init__(self, method_name='runTest'):
        super(SoftmaxClassifierTests, self).__init__(method_name)

    def setUp(self):
        # import the custom classifier
        self.softmax_classifier = SoftmaxClassifier()

    def test_softmax_classifier(self):
        # load dataset
        data, target = load_iris().data, load_iris().target

        # split data in train/test sets
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)

        # standardize columns using normal distribution
        # fit on X_train and not on X_test to avoid Data Leakage
        s = StandardScaler()
        X_train = s.fit_transform(X_train)
        X_test = s.transform(X_test)

        # train on X_train and not on X_test to avoid overfitting
        train_p = self.softmax_classifier.fit_predict(X_train, y_train)
        test_p = self.softmax_classifier.predict(X_test)

        # display precision, recall and f1-score on train/test set
        print("train : " + str(precision_recall_fscore_support(y_train, train_p, average="macro")))
        print("test : " + str(precision_recall_fscore_support(y_test, test_p, average="macro")))

        plt.plot(self.softmax_classifier.losses_)
        plt.show()


if __name__ == '__main__':
    unittest.main()
