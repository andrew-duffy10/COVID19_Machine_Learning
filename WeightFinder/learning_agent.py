
from WeightFinder.data_accumulation import DataAcc


class MultipleRegressionAgent:

    def __init__(self, independent_variables, dependent_variable):
        self.X = independent_variables
        self.y = dependent_variable
        self.data_acc = DataAcc()
        self.variable_weights = None

    def run_for_us_state(self, state, start_day, end_day):
        """

        :param start_day: The first day of the timeseries to run a regression on
        :param end_day: the last day of the timeseries to run a regression on
        :param state: a US state name. ex: 'Massachusetts', 'Kansas', 'California', ...
        :return: a dictionary containing the regressed coefficients mapped to the variable name
        """
        self.pull_data(start_day, end_day)
        data = self.data_acc.get_state(state)
        data = data.dropna()
        X = data[self.X]
        y = data[self.y]
        if len(X) < 1:
            raise ValueError(f"No data found for {state} over the date range {start_day} to {end_day}.")
        self.variable_weights = self.run_regression(X, y)

        variable_weights = {self.X[i]: self.variable_weights[i] for i in range(len(self.X))}
        return variable_weights

    def run_for_day(self, date):
        """
        Runs a multiple-variable regression across all 50 states on a particular day and finds the variable coefficients
        :param date: the day whose variables will be used to run the regression
        :return: a dictionary containing the regressed coefficients mapped to the variable name
        """
        self.pull_data(date, date)
        data = self.data_acc.get_day(date)
        data = data.dropna()
        X = data[self.X]
        y = data[self.y]
        if len(X) < 1:
            raise ValueError(f"No data found for {date}.")
        self.variable_weights = self.run_regression(X, y)
        out = {self.X[i]: self.variable_weights[i] for i in range(len(self.X))}
        return out

    def pull_data(self, start_day, end_day):
        """
        Pulls the data from github and filters it to the columns needed for this regression
        """
        fields = ['Province_State'] + self.X + [self.y]
        self.data_acc.pull_data(start_day, end_day, fields)

    def run_regression(self,X,y):
        raise NotImplementedError("Do not use this super class to run a regression. Instead, call homebrew_agent or sklearn_agent")
