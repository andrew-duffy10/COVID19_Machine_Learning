from WeightFinder.learning_agent import MultipleRegressionAgent
from sklearn import linear_model


class HomebrewAgent(MultipleRegressionAgent):

    def __init__(self, independent_variables, dependent_variable):
        super().__init__(independent_variables, dependent_variable)

    def run_regression(self, X, y):
        linear_regression = linear_model.LinearRegression()
        linear_regression.fit(X, y)
        return linear_regression.coef_



