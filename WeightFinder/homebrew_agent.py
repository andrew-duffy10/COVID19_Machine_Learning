import numpy as np

from WeightFinder.learning_agent import MultipleRegressionAgent


class HomebrewAgent(MultipleRegressionAgent):
    """
    #TODO: implement
    Implements a multiple regression agent using a homebrewed regression algorithm.
    """

    def __init__(self, independent_variables, dependent_variable):
        super().__init__(independent_variables, dependent_variable)
        self.coefficients = None
        self.intercept = None

    def run_regression(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        X = self._add_intercept(X)
        weights = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)
        self.intercept = weights[0]
        self.coefficients = weights[1:]
        return self.coefficients

    @staticmethod
    def _add_intercept(X):
        """

        :param X:
        :return:
        """
        ones = np.ones(shape=X.shape[0]).reshape(-1, 1)
        return np.concatenate((ones, X), 1)

    def predict(self, entry):
        """

        :param entry:
        :return:
        """
        prediction = self.intercept
        for value, coeff in zip(entry, self.coefficients):
            prediction += (value * coeff)
        return prediction
