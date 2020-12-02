from WeightFinder.learning_agent import MultipleRegressionAgent
from sklearn import linear_model


class HomebrewAgent(MultipleRegressionAgent):
    """
    #TODO: implement
    Implements a multiple regression agent using a homebrewed regression algorithm.
    """
    def __init__(self, independent_variables, dependent_variable):
        super().__init__(independent_variables, dependent_variable)

    def run_regression(self, X, y):
        """
        #TODO: implement
        Runs a multiple regression over the independent variables in X to see their weighted relationship to the
        dependent variable in y.

        :param X: A List of colums (variables) whose weights will be found with respect to y
        :param y: A column (variable) that is dependent on the variables in X
        :return: The list of coefficients calculated by the regression algorithm
        """
        linear_regression = linear_model.LinearRegression()
        linear_regression.fit(X, y)
        print(linear_regression.predict([[3000,2000]]))

        return linear_regression.coef_



