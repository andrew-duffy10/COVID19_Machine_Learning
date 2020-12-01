from WeightFinder.learning_agent import MultipleRegressionAgent
from sklearn import linear_model


class SklearnAgent(MultipleRegressionAgent):
    """
    Implements a multiple regression agent using the sklearn library's regression functions.
    """
    def __init__(self, independent_variables, dependent_variable):
        super().__init__(independent_variables, dependent_variable)

    def run_regression(self, X, y):
        """
        Runs a multiple regression over the independent variables in X to see their weighted relationship to the
        dependent variable in y.
        :param X: A List of column names (variables) whose weights will be found
        :param y: A column name (variable) that is dependent on the variables in X
        :return: The list of coefficients calculated by the regression algorithm
        """
        linear_regression = linear_model.LinearRegression()
        linear_regression.fit(X, y)
        return linear_regression.coef_



