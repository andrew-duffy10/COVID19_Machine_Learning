from WeightFinder.learning_agent import MultipleRegressionAgent
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np


class SklearnAgentNN(MultipleRegressionAgent):
    """
    Implements a nueral net regression agent using the sklearn library's regression functions.
    """
    def __init__(self, independent_variables, dependent_variable):
        super().__init__(independent_variables, dependent_variable)
        self.regr = None

    def run_regression(self, X, y):
        """
        Runs a nueral net regression over the independent variables in X to see their weighted relationship to the
        dependent variable in y.

        :param X: A List of colums (variables) whose weights will be found with respect to y
        :param y: A column (variable) that is dependent on the variables in X
        :return: The list of coefficients calculated by the regression algorithm
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
        self.regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)

        return self.regr.score(X_test, y_test)

    def predict(self,entry:list):
        """

        :param entry:
        :return:
        """
        print(entry)
        print(self.regr.predict(np.array(entry).reshape(-1,1)))
        return self.regr.predict(np.array(entry).reshape(-1,1))
