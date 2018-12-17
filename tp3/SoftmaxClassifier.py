# ----------------------------------------------------------------
# Authors: Mathieu Kaboré, Florence Gaborit and Reph D. Mombrun
# Date: 12/09/2018
# Last update: 02/11/2018
# INF8215 TP3
# ----------------------------------------------------------------

from sklearn.base import BaseEstimator, ClassifierMixin

import sklearn.preprocessing
import numpy as np


class SoftmaxClassifier(BaseEstimator, ClassifierMixin):
    """A softmax classifier"""

    def __init__(self, lr=0.1, alpha=100, n_epochs=1000, eps=1.0e-5, threshold=1.0e-10, regularization=True,
                 early_stopping=True):

        """
            self.lr : the learning rate for weights update during gradient descent
            self.alpha: the regularization coefficient 
            self.n_epochs: the number of iterations
            self.eps: the threshold to keep probabilities in range [self.eps;1.-self.eps]
            self.regularization: Enables the regularization, help to prevent overfitting
            self.threshold: Used for early stopping, if the difference between losses during 
                            two consecutive epochs is lower than self.threshold, then we stop the algorithm
            self.early_stopping: enables early stopping to prevent overfitting
        """

        self.lr = lr
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.eps = eps
        self.regularization = regularization
        self.threshold = threshold
        self.early_stopping = early_stopping

        # Initialize the quantity of features, classes and losses
        self.nb_features = 0
        self.nb_classes = 0
        self.losses_ = []

    """
        Public methods, can be called by the user
        To create a custom estimator in sklearn, we need to define the following methods:
        * fit
        * predict
        * predict_proba
        * fit_predict        
        * score
    """

    def fit(self, X, y=None):
        """
            In:
            X : the set of examples of shape nb_example * self.nb_features
            y: the target classes of shape nb_example *  1

            Do:
            Initialize model parameters: self.theta_
            Create X_bias i.e. add a column of 1. to X , for the bias term
            For each epoch
                compute the probabilities
                compute the loss
                compute the gradient
                update the weights
                store the loss
            Test for early stopping

            Out:
            self, in sklearn the fit method returns the object itself
        """
        prev_loss = np.inf
        self.losses_.clear()

        self.nb_features = X.shape[1]
        self.nb_classes = len(np.unique(y))

        # Add a column of 1 to the matrix of examples X(m*n).
        # The resulting new matrix is called X_bias(m*(n+1)).
        X_bias = np.insert(X, self.nb_features, 1, axis=1)

        # Initialize the matrix theta((n+1)*k) with random numbers
        self.theta_ = np.random.rand(self.nb_features + 1, self.nb_classes)

        for epoch in range(self.n_epochs):

            # Compute the logits. The resulting matrix is of size(m*k)
            # z = x * Θ is a vector of size K, i.e. (K*1) that corresponds to the logits
            # related to the example x for each of the K classes.
            # Here, we are computing simultaneously for the m examples.
            logits = np.dot(X_bias, self.theta_)

            # Get the corresponding probabilities of the logits by using softmax
            # Good intro to softmax :
            # https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
            probabilities = self._softmax(logits)

            loss = self._cost_function(probabilities, y)

            # Update theta
            self.theta_ = self.theta_ - (np.multiply(self.lr, self._get_gradient(X_bias, y, probabilities)))

            self.losses_.append(loss)

            if self.early_stopping:
                if len(self.losses_)>0: #éviter une erreur au cas où
                    if self.losses_[-2]-self.losses_[-1]<self.threshold:
                        print("Dernier pas trop petit")
                    break

        return self

    def predict_proba(self, X, y=None):
        """
            In:
            X without bias

            Do:
            Add bias term to X
            Compute the logits for X
            Compute the probabilities using softmax

            Out:
            Predicted probabilities
        """
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        # self = self.fit(X, y)
        # return self.probabilities

    def predict(self, X, y=None):
        """
            In:
            X without bias

            Do:
            Add bias term to X
            Compute the logits for X
            Compute the probabilities using softmax
            Predict the classes

            Out:
            Predicted classes
        """
        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X, y)

    def score(self, X, y=None):  # proablement fuasse, à corriger
        """
            In :
            X set of examples (without bias term)
            y the true labels

            Do:
                predict probabilities for X
                Compute the log loss without the regularization term

            Out:
            log loss between prediction and true labels
        """
        predictions = self.predict_proba(X)
        self.regularization = False
        return self._cost_function(predictions, y)

    """
        Private methods, their names begin with an underscore
    """

    def _cost_function(self, probabilities, y):
        """
            In :
            y without one hot encoding
            probabilities computed with softmax

            Do:
            One-hot encode y
            Ensure that probabilities are not equal to either 0. or 1. using self.eps
            Compute log_loss
            If self.regularization, compute l2 regularization term
            Ensure that probabilities are not equal to either 0. or 1. using self.eps

            Out:
            cost (real number)
        """
        hot_y = self._one_hot(y)
        probabilities[probabilities == 0] = self.eps
        probabilities[probabilities == 1] = 1 - self.eps
        if self.regularization:
            l2 = np.multiply(self.alpha, np.sum(np.square(self.theta_)) - np.sum(
                np.square(self.theta_[0])))  # on enleve la première ligne dans la somme
        else:
            l2 = 0

        m = probabilities.shape[0]
        log_loss = -1 / m * np.sum(np.multiply(hot_y, np.log(probabilities)))
        return log_loss + l2

    @staticmethod
    def _one_hot(y):
        """
            In :
            Target y: nb_examples * 1

            Do:
            One hot-encode y
            [1,1,2,3,1] --> [[1,0,0],
                             [1,0,0],
                             [0,1,0],
                             [0,0,1],
                             [1,0,0]]
            Out:
            y one-hot encoded
        """
        lb = sklearn.preprocessing.LabelBinarizer()
        lb.fit(y)
        return np.array(lb.transform(y))

    @staticmethod
    def _softmax(Z):
        """
            In :
             nb_examples * self.nb_classes
            Do:
            Compute softmax on logits

            Out:
            Probabilities
        """
        # --------------------------------------------------------------------------------------------------
        # Good intro to softmax:
        # https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
        # --------------------------------------------------------------------------------------------------

        # Z is the input matrix, i.e. the logits
        # and z is one row in Z, where z = x ∗ Θ is a vector of size K,
        # i.e. (K*1) that corresponds to the logits related to the example x
        # for each of the K classes.
        # Here, we are computing simultaneously for the m examples, i.e. m rows.
        # Compute the exponential of all the elements in Z(m*k) and store
        # the resulting matrix in exponential (m*k).
        exponential = np.exp(Z)

        # Compute the sum for each z in the exponential matrix.
        # The resulting matrix is of size (m*1) since the elements
        # of all the columns for a specific row are summed up.
        total = np.sum(exponential, axis=1)

        # Inverse all the elements in total
        total = np.divide(np.ones(total.shape), total)

        # Do an element-wise multiplication to finally compute the softmax
        # Good intro to numpy for broadcasting:
        # https://howtothink.readthedocs.io/en/latest/PvL_06.html
        return (exponential.T * total).T

    def _get_gradient(self, X, y, probas):
        """
            In:
            X with bias
            y without one hot encoding
            probabilities resulting of the softmax step

            Do:
            One-hot encode y
            Compute gradients
            If self.regularization add l2 regularization term

            Out:
            Gradient
        """

        # One-hot encode y. Btw, yohe stands for y one hot encoded :)
        yohe = self._one_hot(y)

        # Compute gradients
        m = X.shape[0]  # Get the amount of rows of the X matrix
        grad = np.multiply(1.0 / m, np.dot(np.transpose(X), probas - yohe))

        # If self.regularization add l2 regularization term
        if self.regularization:
            L2 = np.multiply(self.alpha, np.sum(np.square(self.theta_))
                             - np.sum(np.square(self.theta_[0]))) # on enleve la première ligne dans la somme
            grad = grad + L2

        return grad  # voir si on copie la valeur sur un matrice..? et enlever la dernière colonne (on ne dérive le biais ce qui donne zero)
